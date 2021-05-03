//
//  Configuration.swift
//  MeatModeler
//
//  Created by Kevin Grogan on 4/17/21.
//

import Foundation

typealias SpatialTemperature = (x: Float, T: Float)

class Controller: ObservableObject {
    @Published var startTemperature: Float
    @Published var ambientTemperature: Float
    @Published var mass: Float
    @Published var cookTime: Float
    @Published var size: Float
    @Published var temperatureUnit: String
    @Published var timeUnit: String
    @Published var massUnit: String
    @Published var lengthUnit: String
    
    let porkTenderloinModel = PorkTenderloinModel(modelFileInfo: FileInfo(name: "pork_tenderloin", extension: "tflite"))
    
    
    init() {
        startTemperature = 45
        ambientTemperature = 350
        mass = 24
        cookTime = 20
        size = 2
        temperatureUnit = "°F"
        timeUnit = "min"
        massUnit = "oz"
        lengthUnit = "in"
    }
    
    var spatialTemperatures: [SpatialTemperature] {
        do {
            let SIResults = porkTenderloinModel!.computeFinalSpatialTemperature(
                T0: try UnitConverter.converToSI(value: startTemperature, from: temperatureUnit),
                TInfty: try UnitConverter.converToSI(value: ambientTemperature, from: temperatureUnit),
                D: try UnitConverter.converToSI(value: size, from: lengthUnit),
                tFinal: try UnitConverter.converToSI(value: cookTime, from: timeUnit))!
            return try SIResults.map{
                SpatialTemperature(
                    x: try UnitConverter.convertFromSI(value: $0.x, to: lengthUnit),
                    T: try UnitConverter.convertFromSI(value: $0.T, to: temperatureUnit)
                )
            }
            
        }
        catch {
            print("Computation of spatial temperature failed. Defaulting to linear distribution.")
            return [
                SpatialTemperature(x: 0.0, T: startTemperature),
                SpatialTemperature(x: size, T: ambientTemperature)
            ]
        }
    }
    
    var temperatureDistribution: [TemperatureBin] {
        PorkTenderloinModel.computeFinalTemperatureDistribution(spatialTemperatures, numBins: 4)
    }
    
}

struct UnitConverter {
    static let temperatureUnits = [
        "°F": UnitTemperature.fahrenheit,
        "°C": UnitTemperature.celsius,
        "K": UnitTemperature.kelvin
    ]
    static let timeUnits = [
        "s": UnitDuration.seconds,
        "min": UnitDuration.minutes,
        "hr": UnitDuration.hours
    ]
    static let massUnits = [
        "oz": UnitMass.ounces,
        "lb": UnitMass.pounds,
        "kg": UnitMass.kilograms
    ]
    static let lengthUnits = [
        "in": UnitLength.inches,
        "cm": UnitLength.centimeters,
        "m": UnitLength.meters
    ]
    
    static func converToSI(value: Float, from unitName: String) throws -> Float {
        let unitPair = try getUnitPair(from: unitName)
        var measurement = Measurement(value: Double(value), unit: unitPair.unit)
        measurement.convert(to: unitPair.SIUnit)
        return Float(measurement.value)
    }
    
    private static func getUnitPair(from unitName: String) throws -> (unit: Dimension, SIUnit: Dimension) {
        // Returns the unit for the provided unit name and the corresponding SI unit.
        var unit: Dimension
        var SIUnit: Dimension
        if temperatureUnits.keys.contains(unitName) {
            unit =  temperatureUnits[unitName]!
            SIUnit = UnitTemperature.kelvin
        }
        else if timeUnits.keys.contains(unitName) {
            unit =  timeUnits[unitName]!
            SIUnit = UnitDuration.seconds
        }
        else if massUnits.keys.contains(unitName) {
            unit =  massUnits[unitName]!
            SIUnit = UnitMass.kilograms
        }
        else if lengthUnits.keys.contains(unitName) {
            unit =  lengthUnits[unitName]!
            SIUnit = UnitLength.meters
        }
        else {
            throw "Unexpected unit name \(unitName)."
        }
        return (unit: unit, SIUnit: SIUnit)
    }
    
    static func convertFromSI(value: Float, to unitName: String) throws -> Float {
        let unitPair = try getUnitPair(from: unitName)
        var measurement = Measurement(value: Double(value), unit: unitPair.SIUnit)
        measurement.convert(to: unitPair.unit)
        return Float(measurement.value)
    }
}

extension String: Error {}  // Simple errors to print a message
