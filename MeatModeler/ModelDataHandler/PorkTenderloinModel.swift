//
//  ModelDataHandler.swift
//  MeatModeler. Adapted from https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios
//
//  Created by Kevin Grogan on 4/24/21.
//

import TensorFlowLite
import Foundation


/// An inference from invoking the `Interpreter`.
struct Prediction {
    let theta: Float
    let rho: Float
    let tau: Float
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)


/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class PorkTenderloinModel {

  // MARK: - Internal Properties

  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int

  let resultCount = 3
  let threadCountLimit = 10

  // MARK: - Private Properties

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

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

    
    func predict(kappa: Float, theta0: Float, nusseltNumber: Float) -> [Prediction]? {

        let outputTensor: Tensor
        do {
            let inputData = Data(copyingBufferOf: [kappa, theta0, nusseltNumber])
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
        
        // Unravel the predictions
        let predictions = self.getPredictions(from: outputs)
        return predictions
  }
    
    private func getPredictions(from outputs: [Float]) -> [Prediction] {
        var predictions = [Prediction]()
        let numOutputs = outputs.count
        let dim = sqrt(Float(numOutputs))  // dimension the same in all directions
        assert(dim.truncatingRemainder(dividingBy: 1.0) == 0)
        let delta = 1 / (dim - 1)  // rho and tau in [0, 1]
        for (index, theta) in outputs.enumerated() {
            let rho = Float(index % 3) * delta
            let tau = Float(index / 3) * delta
            predictions.append(Prediction(theta: theta, rho: rho, tau: tau))
        }
        return predictions
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
