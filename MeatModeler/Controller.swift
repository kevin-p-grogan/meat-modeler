//
//  Configuration.swift
//  MeatModeler
//
//  Created by Kevin Grogan on 4/17/21.
//

import Foundation

typealias SpatialTemperatureDatum = (x: Float, T: Float)

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
        cookTime = 1200
        size = 2
        temperatureUnit = UnitConverter.temperatureUnits.keys.first ?? "°F"
        timeUnit = UnitConverter.timeUnits.keys.first ?? "s"
        massUnit = UnitConverter.massUnits.keys.first ?? "oz"
        lengthUnit = UnitConverter.lengthUnits.keys.first ?? "in"
    }
    
    var spatialTemperatureData: [SpatialTemperatureDatum] {
        do {
            return porkTenderloinModel!.computeSpatialTemperature(
                T0: try UnitConverter.converToSI(value: startTemperature, unitName: temperatureUnit),
                TInfty: try UnitConverter.converToSI(value: ambientTemperature, unitName: temperatureUnit),
                D: try UnitConverter.converToSI(value: size, unitName: lengthUnit),
                tFinal: try UnitConverter.converToSI(value: cookTime, unitName: temperatureUnit))!
        }
        catch {
            print("Computation of spatial temperature failed. Defaulting to linear distribution.")
            return [
                SpatialTemperatureDatum(x: 0.0, T: startTemperature),
                SpatialTemperatureDatum(x: size, T: ambientTemperature)
            ]
        }
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
    
    static func converToSI(value: Float, unitName: String) throws -> Float {
        var currentUnit: Dimension
        var finalUnit: Dimension
        if temperatureUnits.keys.contains(unitName) {
            currentUnit =  temperatureUnits[unitName]!
            finalUnit = UnitTemperature.kelvin
        }
        else if timeUnits.keys.contains(unitName) {
            currentUnit =  timeUnits[unitName]!
            finalUnit = UnitDuration.seconds
        }
        else if massUnits.keys.contains(unitName) {
            currentUnit =  massUnits[unitName]!
            finalUnit = UnitMass.kilograms
        }
        else if lengthUnits.keys.contains(unitName) {
            currentUnit =  lengthUnits[unitName]!
            finalUnit = UnitLength.meters
        }
        else {
            throw "Unexpected unit name \(unitName)."
        }
        var measurement = Measurement(value: Double(value), unit: currentUnit)
        measurement.convert(to: finalUnit)
        return Float(measurement.value)
    }
}

extension String: Error {}  // Simple errors to print a message
