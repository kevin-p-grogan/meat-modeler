//
//  ContentView.swift
//  MeatModeler
//
//  Created by Kevin Grogan on 4/17/21.
//

import SwiftUI

struct MainView: View {
    @ObservedObject var control = Controller()
    var body: some View {
        TabView {
            ConfigurationView(control: control)
                .tabItem {
                    Image(systemName: "slider.horizontal.3")
                }
            ResultsView(control: control)
                .tabItem {
                    Image(systemName:"waveform.path.ecg")
                }
        }
        .font(.headline)
        .preferredColorScheme(.dark)
        .accentColor(.red)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
    }
}
