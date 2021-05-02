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
        VStack {
            LineChart(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData))
//                    .pointMarkers(chartData: data)
                        .touchOverlay(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData), specifier: "%.0f")
//                    .yAxisPOI(chartData: data,
//                              markerName: "Step Count Aim",
//                              markerValue: 15_000,
//                              labelPosition: .center(specifier: "%.0f"),
//                              labelColour: Color.black,
//                              labelBackground: Color(red: 1.0, green: 0.75, blue: 0.25),
//                              lineColour: Color(red: 1.0, green: 0.75, blue: 0.25),
//                              strokeStyle: StrokeStyle(lineWidth: 3, dash: [5,10]))
//                    .yAxisPOI(chartData: data,
//                              markerName: "Minimum Recommended",
//                              markerValue: 10_000,
//                              labelPosition: .center(specifier: "%.0f"),
//                              labelColour: Color.white,
//                              labelBackground: Color(red: 0.25, green: 0.75, blue: 1.0),
//                              lineColour: Color(red: 0.25, green: 0.75, blue: 1.0),
//                              strokeStyle: StrokeStyle(lineWidth: 3, dash: [5,10]))
//                    .averageLine(chartData: data,
//                                 strokeStyle: StrokeStyle(lineWidth: 3, dash: [5,10]))
                        .xAxisGrid(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData))
                        .yAxisGrid(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData))
                        .xAxisLabels(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData))
                        .yAxisLabels(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData))
                        .infoBox(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData))
                        .headerBox(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData))
                        .legends(chartData: ResultsView.spatialTemperatureChartData(control.spatialTemperatureData), columns: [GridItem(.flexible()), GridItem(.flexible())])
                        .id(ResultsView.spatialTemperatureChartData(control.spatialTemperatureData).id)
                    .frame(minWidth: 150, maxWidth: 900, minHeight: 150, idealHeight: 250, maxHeight: 400, alignment: .center)
                    .padding()
            }
            .navigationTitle("Week of Data")
        }
        
    static func spatialTemperatureChartData(_ spatialTemperatureData: [SpatialTemperatureDatum]) -> LineChartData {
        let data = LineDataSet(dataPoints: spatialTemperatureData.map{LineChartDataPoint(value: Double($0.T))},
//                LineChartDataPoint(value: 12000.0, xAxisLabel: "M", description: "Monday"),
//                LineChartDataPoint(value: 10000, xAxisLabel: "T", description: "Tuesday"),
//                LineChartDataPoint(value: 8000,  xAxisLabel: "W", description: "Wednesday"),
//                LineChartDataPoint(value: 17500, xAxisLabel: "T", description: "Thursday"),
//                LineChartDataPoint(value: 16000, xAxisLabel: "F", description: "Friday"),
//                LineChartDataPoint(value: 11000, xAxisLabel: "S", description: "Saturday"),
//                LineChartDataPoint(value: 9000,  xAxisLabel: "S", description: "Sunday")
//            ],
            legendTitle: "Steps",
            pointStyle: PointStyle(),
            style: LineStyle(lineColour: ColourStyle(colour: .red), lineType: .curvedLine))
            
            let metadata   = ChartMetadata(title: "Spatial Temperature") //, subtitle: "Over a Week")
            
            let gridStyle  = GridStyle(numberOfLines: 7,
                                       lineColour   : Color(.lightGray).opacity(0.5),
                                       lineWidth    : 1,
                                       dash         : [8],
                                       dashPhase    : 0)
            
            let chartStyle = LineChartStyle(infoBoxPlacement    : .infoBox(isStatic: false),
                                            infoBoxBorderColour : Color.primary,
                                            infoBoxBorderStyle  : StrokeStyle(lineWidth: 1),
                                            
                                            markerType          : .vertical(attachment: .line(dot: .style(DotStyle()))),
                                            
                                            xAxisGridStyle      : gridStyle,
                                            xAxisLabelPosition  : .bottom,
                                            xAxisLabelColour    : Color.primary,
                                            xAxisLabelsFrom     : .dataPoint(rotation: .degrees(0)),
                                            
                                            yAxisGridStyle      : gridStyle,
                                            yAxisLabelPosition  : .leading,
                                            yAxisLabelColour    : Color.primary,
                                            yAxisNumberOfLabels : 7,
                                            
//                                            baseline            : .minimumWithMaximum(of: 5000),
//                                            topLine             : .maximum(of: 10),
                                            
                                            globalAnimation     : .easeOut(duration: 1))
            
            return LineChartData(dataSets       : data,
                                 metadata       : metadata,
                                 chartStyle     : chartStyle)
            
        }
    }

struct ResultsView_Previews: PreviewProvider {
    static var previews: some View {
        ResultsView(control: Controller())
    }
}
