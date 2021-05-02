//
//  MeatModelerTests.swift
//  MeatModelerTests
//
//  Created by Kevin Grogan on 4/17/21.
//

import XCTest
@testable import MeatModeler

class MeatModelerTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testTFLiteModel() throws {
        // Tests the predictions from tensorflow on a test model.
        let modelInfo = FileInfo(name: "test_model", extension: "tflite")
        let model =  PorkTenderloinModel(modelFileInfo: modelInfo, shouldWeightRhoByArea: false)!
        let a: Float = -1.0
        let b: Float = 3.0
        let predictions = model.makeNondimensionalPredictions(kappa: a, theta0: b)!
        for prediction in predictions {
            // Compare against the test model. See python unit test for definition.
            let testValue = a * prediction.rho + b * prediction.tau
            XCTAssertEqual(prediction.theta, testValue, accuracy: 1e-3)
        }
    }
    
    func testPorkTenderloinModel() throws {
        // Tests the predictions from tensorflow on a test model.
        let modelInfo = FileInfo(name: "pork_tenderloin", extension: "tflite")
        let model =  PorkTenderloinModel(modelFileInfo: modelInfo)!
        let predictions = model.makeNondimensionalPredictions(kappa: 1.0, theta0: -1.0)!
        // assertions about the bounds of the model
        let rhos = predictions.map{$0.rho}
        XCTAssertEqual(rhos.max()!, 0.5, accuracy: 1e-3)
        XCTAssertEqual(rhos.min()!, 0.0, accuracy: 1e-3)
        let taus = predictions.map{$0.tau}
        XCTAssertEqual(taus.max()!, 1.0, accuracy: 1e-3)
        XCTAssertEqual(taus.min()!, 0.0, accuracy: 1e-3)
        // now look at the spatial predictions
        let spatialTemperatures = model.computeSpatialTemperature(T0: 300, TInfty: 400, D: 1.0, tFinal: 1000)!
        XCTAssertTrue(spatialTemperatures.count > 0)
    }
    
    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
