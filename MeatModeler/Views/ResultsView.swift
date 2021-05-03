//
//  ResultsView.swift
//  MeatModeler
//
//  Created by Kevin Grogan on 4/17/21.
//

import SwiftUI
import SwiftUICharts


struct ResultsView: View {
    @ObservedObject var control: Controller
    
    
    var body: some View {
        NavigationView{
            Form {
                Section{
                    BarChart(chartData: temperatureDistributionChartData)
                        .xAxisLabels(chartData: temperatureDistributionChartData)
                        .headerBox(chartData: temperatureDistributionChartData)
                        .id(temperatureDistributionChartData.id)
                        .frame(minWidth: 150, maxWidth: 900, minHeight: 150, idealHeight: 250, maxHeight: 600, alignment: .center)
                }
                Section{
                    ResultsTableView(control: control)
                        .frame(minWidth: 150, maxWidth: 900, minHeight: 150, idealHeight: 250, maxHeight: 300, alignment: .center)
                    }
                }.navigationBarTitle(Text("Results"))
        }
        }
        
    var temperatureDistributionChartData: BarChartData {
        let data = BarDataSet(dataPoints:
                                control.temperatureDistribution.map{BarChartDataPoint(
                                value: Double($0.probability),
                                xAxisLabel: String(format: "%.0f \(control.temperatureUnit)", $0.T))
                                })
                
        let barStyle = BarStyle(
            barWidth: 0.9
        )
        
        
        return BarChartData(dataSets: data, barStyle: barStyle)
        
    }
    
    func formatFloatLabel(label: Float) -> String {
        return String(format: "%.0f \(control.temperatureUnit)", label)
    }
}

struct ResultsTableView: View {
    @ObservedObject var control: Controller
    
    let horizontalSpacing: CGFloat = 50
    let verticalSpacing: CGFloat = 30
    
    var body: some View {
        HStack(spacing: horizontalSpacing){
                VStack(spacing: verticalSpacing){
                    Spacer()
                    Text("Minimum").font(.headline)
                    Spacer()
                    Text("Maximum").font(.headline)
                    Spacer()
                    Text("Mean").font(.headline)
                    Spacer()
                }
                Divider()
                VStack(spacing: verticalSpacing){
                    Spacer()
                    Text(String(format: "%.0f \(control.temperatureUnit)", control.spatialTemperatures.map{$0.T}.min() ?? control.startTemperature))
                    Spacer()
                    Text(String(format: "%.0f \(control.temperatureUnit)", control.spatialTemperatures.map{$0.T}.max() ?? control.ambientTemperature))
                    Spacer()
                    Text(String(format: "%.0f \(control.temperatureUnit)", control.spatialTemperatures.map{$0.T}.mean))
                    Spacer()
                }

            }
    }
}


struct ResultsView_Previews: PreviewProvider {
    static var previews: some View {
        ResultsView(control: Controller())
    }
}
