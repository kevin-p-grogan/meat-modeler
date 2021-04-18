//
//  ContentView.swift
//  MeatModeler
//
//  Created by Kevin Grogan on 4/17/21.
//

import SwiftUI

struct MainView: View {
    @ObservedObject var config = Configuration()
    var body: some View {
        TabView {
            ConfigurationView(config: config)
                .tabItem {
                    Image(systemName: "slider.horizontal.3")
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
