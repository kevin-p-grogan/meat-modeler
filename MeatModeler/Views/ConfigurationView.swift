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
                    TemperatureConfigurationView(control: control)
                }
                Section{
                    MassConfigurationView(control: control)
                }
                Section{
                    TimeConfigurationView(control: control)
                }
                Section{
                    SizeConfigurationView(control: control)
                }
            }.navigationBarTitle("Configure")
        }
    }
}

struct TemperatureConfigurationView: View {
    @ObservedObject var control: Controller
    var body: some View {
        Text("Temperature").font(.headline)
        HStack{
            Text("Start").frame(maxWidth: .infinity)
            Divider().frame(maxWidth: .infinity)
            TextField("", value: $control.startTemperature, formatter: formatter).frame(maxWidth: .infinity)
        }
        HStack{
            Text("Ambient").frame(maxWidth: .infinity)
            Divider().frame(maxWidth: .infinity)
            TextField("", value: $control.ambientTemperature, formatter: formatter)
        }
        Picker("Temperature Unit", selection: $control.temperatureUnit) {
            ForEach(Array(UnitConverter.temperatureUnits.keys), id:\.self) { tu in
                Text(tu)
            }
        }.pickerStyle(SegmentedPickerStyle())
    }
}

struct MassConfigurationView: View {
    @ObservedObject var control: Controller
    var body: some View {
        HStack(spacing: 58.0){
            Text("Mass").font(.headline)
            Divider()
            TextField("", value: $control.mass, formatter: formatter)
        }
        Picker("Mass Unit", selection: $control.massUnit) {
            ForEach(Array(UnitConverter.massUnits.keys), id:\.self) { mu in
                Text(mu)
            }
        }.pickerStyle(SegmentedPickerStyle())
    }
}

struct TimeConfigurationView: View {
    @ObservedObject var control: Controller
    var body: some View {
        HStack(spacing: 58.0){
            Text("Time").font(.headline)
            Divider()
            TextField("", value: $control.cookTime, formatter: formatter)
        }
        Picker("Time Unit", selection: $control.timeUnit) {
            ForEach(Array(UnitConverter.timeUnits.keys), id:\.self) { tu in
                Text(tu)
            }
        }.pickerStyle(SegmentedPickerStyle())
    }
}

struct SizeConfigurationView: View {
    @ObservedObject var control: Controller
    var body: some View {
        HStack(spacing: 58.0){
            Text("Size").font(.headline)
            Divider()
            TextField("", value: $control.size, formatter: formatter)
        }
        Picker("Lenght Unit", selection: $control.lengthUnit) {
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
