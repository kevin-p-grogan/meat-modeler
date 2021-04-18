//
//  ResultsView.swift
//  MeatModeler
//
//  Created by Kevin Grogan on 4/17/21.
//

import SwiftUI
import SwiftUICharts


struct ResultsView: View {
    @ObservedObject var config: Configuration
    
    
    var body: some View {
        VStack{
            LineView(data: [8,23,54,32,12,37,7,23,43], title: "Axial Temperature") // legend is optional,
                .padding()
            LineView(data: [8,23,54,32,12,37,7,23,430], title: "Radial Temperature") // legend is optional,
                .padding()
        }
    }
}

struct ResultsView_Previews: PreviewProvider {
    static var previews: some View {
        ResultsView(config: Configuration())
    }
}
