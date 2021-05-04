//
//  ModelDataHandler.swift
//  MeatModeler. Adapted from https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios
//
//  Created by Kevin Grogan on 4/24/21.
//

import TensorFlowLite
import Foundation


/// An inference from invoking the `Interpreter`.
struct NondimensionalPredictions {
    let theta: Float
    let rho: Float
    let tau: Float
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Alias for binning temperatures
typealias TemperatureBin = (T: Float, probability: Float)


/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class PorkTenderloinModel {

    // MARK: - Internal Properties

    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int

    let resultCount = 3
    let threadCountLimit = 10

    // Model parameters
    let rhoMin: Float = 0.0
    let rhoMax: Float = 0.5
    let tauMin: Float = 0.0
    let tauMax: Float = 1.0
    let shouldWeightRhoByArea: Bool
    let k: Float = 0.47  // [W/m.K] src: https://www.engineeringtoolbox.com/food-thermal-conductivity-d_2177.html
    let C: Float = 2760  // [J/kg.K] src: https://www.engineeringtoolbox.com/specific-heat-capacity-food-d_295.html
    let density: Float = 1090 // [kg/m^3]
    
    let tauEps: Float = 1e-6

    // MARK: - Private Properties

    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter

    // MARK: - Initialization

    /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
    init?(modelFileInfo: FileInfo, threadCount: Int = 1, shouldWeightRhoByArea: Bool = true) {
        let modelFilename = modelFileInfo.name

        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        self.shouldWeightRhoByArea = shouldWeightRhoByArea

        // Specify the options for the `Interpreter`.
        self.threadCount = threadCount
        var options = Interpreter.Options()
        options.threadCount = threadCount
        do {
          // Create the `Interpreter`.
          interpreter = try Interpreter(modelPath: modelPath, options: options)
          // Allocate memory for the model's input `Tensor`s.
          try interpreter.allocateTensors()
        } catch let error {
          print("Failed to create the interpreter with error: \(error.localizedDescription)")
          return nil
        }
  }

  // MARK: - Internal Methods

    
    func makeNondimensionalPredictions(kappa: Float, theta0: Float) -> [NondimensionalPredictions]? {

        let outputTensor: Tensor
        do {
            let inputData = Data(copyingBufferOf: [kappa, theta0])
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            outputTensor = try interpreter.output(at: 0)
        }
        catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }

        let outputs: [Float]
        switch outputTensor.dataType {
        case .float32:
          outputs = [Float](unsafeData: outputTensor.data) ?? []
        default:
          print("Output tensor data type \(outputTensor.dataType) is unsupported for this example app.")
          return nil
        }
        
        let cleanedOutputs = cleanOutputs(outputs)
        
        // Unravel the predictions
        let nondimensionalPredictions = self.associateSpaceAndTime(from: cleanedOutputs)
        return nondimensionalPredictions
    }
    
    private func cleanOutputs(_ outputs: [Float]) -> [Float] {
        // Suppress NaNs by filling in zeros
        let outputsWithoutNaN = outputs.map{$0.isNaN ? 0.0: $0}
        return outputsWithoutNaN
    }
    
    private func associateSpaceAndTime(from outputs: [Float]) -> [NondimensionalPredictions] {
        // Assoctiates the spatial and temporal (i.e., rho and tau, respectively) with the temperature (theta).
        var predictions = [NondimensionalPredictions]()
        let numOutputs = outputs.count
        let dim = sqrt(Float(numOutputs))  // dimension the same in all directions
        assert(dim.truncatingRemainder(dividingBy: 1.0) == 0)
        let rhos = getDiscretization(from: rhoMin, to: rhoMax, withCount: Int(dim), shouldWeightByArea: shouldWeightRhoByArea)
        let taus = getDiscretization(from: tauMin, to: tauMax, withCount: Int(dim))
        for (index, theta) in outputs.enumerated() {
            let rho = rhos[index % Int(dim)]
            let tau = taus[index / Int(dim)]
            predictions.append(NondimensionalPredictions(theta: theta, rho: rho, tau: tau))
        }
        return predictions
    }
    
    private func getDiscretization(from start: Float, to finish: Float, withCount count: Int, shouldWeightByArea: Bool = false) -> [Float] {
        var discretization = [start]
        let range = 0..<count-1
        if shouldWeightByArea {
            let delta = (pow(finish, 2.0) - pow(start, 2.0)) / (Float(count) - 1)
            for index in range {
                discretization.append(sqrt(pow(discretization[index], 2.0) + delta))
            }
        }
        else {
            let delta = (finish - start) / (Float(count) - 1)
            for index in range {
                discretization.append(discretization[index] + delta)
            }
        }
        return discretization
    }
    
    func computeFinalSpatialTemperature(T0: Float, TInfty: Float, D: Float, tFinal: Float) -> [SpatialTemperature]? {
        let kappa = k * tFinal / (density * C * pow(D, 2.0))
        let theta0 = T0/TInfty - 1.0
        let predictions = makeNondimensionalPredictions(kappa: kappa, theta0: theta0)
        let endTimePredictions = predictions?.filter{abs($0.tau-tauMax) <= tauEps}.sorted{$0.rho < $1.rho}
        let dimensionalPredictions = endTimePredictions?.map{SpatialTemperature(x: $0.rho*D, T: TInfty*($0.theta+1))}
        return dimensionalPredictions
    }
        
    static func computeFinalTemperatureDistribution(_ spatialTemperatures: [SpatialTemperature], numBins: Int = 10) -> [TemperatureBin] {
        // Returns the temperature distrubtion sorted by temperature
        let temperatures = spatialTemperatures.map{$0.T}
        return temperatures.bin(numBins: numBins).map{TemperatureBin(T: $0.value, probability: $0.probability)}
    }
}
// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
    /// Creates a new array from the bytes of the given unsafe data.
    ///
    /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
    ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
    ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
    /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
    ///     `MemoryLayout<Element>.stride`.
    /// - Parameter unsafeData: The data containing the bytes to turn into an array.
    init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
    }
}

typealias Bin = (value: Float, probability: Float)

extension Array where Element: BinaryFloatingPoint {
    var mean: Float {
        if self.isEmpty {
            return 0.0
        }
        else {
            let sum = self.reduce(0, +)
            return Float(sum) / Float(self.count)
        }
    }
    
    func bin(numBins: Int) -> [Bin] {
        // Creates bins with a probability given by the frequency of data in the bin and a value equal to the bin's mean.
        if self.isEmpty{
            return []
        }
        else {
            let _numBins = Swift.min(self.count, numBins)
            let min = Float(self.min()!)
            let max = Float(self.max()!)
            let delta = (max-min) / Float(_numBins)
            var bins = Array<Bin>()
            for iBin in 0..<_numBins {
                let start = min + Float(iBin) * delta
                let finish = start + delta
                let filteredValues = self.filter{(start<=Float($0)) && (Float($0)<finish)}
                bins.append(Bin(value: filteredValues.mean,
                                probability: Float(filteredValues.count) / Float(self.count))
                )
            }
            return bins
        }
    }
}
