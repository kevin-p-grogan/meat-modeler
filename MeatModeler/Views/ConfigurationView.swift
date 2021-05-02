//
//  ConfigurationView.swift
//  MeatModeler
//
//  Created by Kevin Grogan on 4/17/21.
//

import SwiftUI

let formatter: NumberFormatter = {
    let formatter = NumberFormatter()
    formatter.numberStyle = .decimal
    formatter.minimumFractionDigits = 0
    formatter.maximumFractionDigits = 1
    return formatter
}()


struct ConfigurationView: View {
    @ObservedObject var control: Controller
    var body: some View {
        NavigationView {
            Form {
                Section{
                    TemperatureConfigurationView(config: control)
                }
                Section{
                    MassConfigurationView(config: control)
                }
                Section{
                    TimeConfigurationView(config: control)
                }
                Section{
                    SizeConfigurationView(config: control)
                }
            }.navigationBarTitle("Configure")
        }
    }
}

struct TemperatureConfigurationView: View {
    @ObservedObject var config: Controller
    var body: some View {
        Text("Temperature").font(.headline)
        HStack{
            Text("Start").frame(maxWidth: .infinity)
            Divider().frame(maxWidth: .infinity)
            TextField("", value: $config.startTemperature, formatter: formatter).frame(maxWidth: .infinity)
        }
        HStack{
            Text("Ambient").frame(maxWidth: .infinity)
            Divider().frame(maxWidth: .infinity)
            TextField("", value: $config.ambientTemperature, formatter: formatter)
        }
        Picker("Temperature Unit", selection: $config.temperatureUnit) {
            ForEach(Array(UnitConverter.temperatureUnits.keys), id:\.self) { tu in
                Text(tu)
            }
        }.pickerStyle(SegmentedPickerStyle())
    }
}

struct MassConfigurationView: View {
    @ObservedObject var config: Controller
    var body: some View {
        HStack(spacing: 58.0){
            Text("Mass").font(.headline)
            Divider()
            TextField("", value: $config.mass, formatter: formatter)
        }
        Picker("Mass Unit", selection: $config.massUnit) {
            ForEach(Array(UnitConverter.massUnits.keys), id:\.self) { mu in
                Text(mu)
            }
        }.pickerStyle(SegmentedPickerStyle())
    }
}

struct TimeConfigurationView: View {
    @ObservedObject var config: Controller
    var body: some View {
        HStack(spacing: 58.0){
            Text("Time").font(.headline)
            Divider()
            TextField("", value: $config.cookTime, formatter: formatter)
        }
        Picker("Time Unit", selection: $config.timeUnit) {
            ForEach(Array(UnitConverter.timeUnits.keys), id:\.self) { tu in
                Text(tu)
            }
        }.pickerStyle(SegmentedPickerStyle())
    }
}

struct SizeConfigurationView: View {
    @ObservedObject var config: Controller
    var body: some View {
        HStack(spacing: 58.0){
            Text("Size").font(.headline)
            Divider()
            TextField("", value: $config.size, formatter: formatter)
        }
        Picker("Lenght Unit", selection: $config.lengthUnit) {
            ForEach(Array(UnitConverter.lengthUnits.keys), id:\.self) { lu in
                Text(lu)
            }
        }.pickerStyle(SegmentedPickerStyle())
    }
}

struct ConfigurationView_Previews: PreviewProvider {
    static var previews: some View {
        ConfigurationView(control: Controller())
    }
}
