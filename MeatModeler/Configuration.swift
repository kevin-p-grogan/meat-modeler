//
//  Configuration.swift
//  MeatModeler
//
//  Created by Kevin Grogan on 4/17/21.
//

import Foundation

class Configuration: ObservableObject {
    @Published var startTemperature: Double
    @Published var finalTemperature: Double
    @Published var mass: Double
    @Published var cookTime: Double
    @Published var size: Double
    @Published var temperatureUnit: String
    @Published var timeUnit: String
    @Published var massUnit: String
    @Published var lengthUnit: String

    
    init() {
        startTemperature = 45
        finalTemperature = 145
        mass = 24
        cookTime = 20
        size = 12
        temperatureUnit = temperatureUnits[0]
        timeUnit = timeUnits[1]
        massUnit = massUnits[0]
        lengthUnit = lengthUnits[0]
    }
}

let temperatureUnits = ["°F", "°C", "K"]
let timeUnits = ["s", "min", "hr"]
let massUnits = ["oz", "lb", "kg"]
let lengthUnits = ["in", "cm", "m"]
