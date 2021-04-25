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
        let model =  PorkTenderloinModel(modelFileInfo: modelInfo)!
        let a: Float = -1.0
        let b: Float = 3.0
        let c: Float = 2.0
        let predictions = model.predict(kappa: a, theta0: b, nusseltNumber: c)!
        for prediction in predictions {
            // Compare against the test model. See python unit test for definition.
            let testValue = a * prediction.rho + b * prediction.tau + c
            XCTAssertEqual(prediction.theta, testValue, accuracy: 1e-3)
        }
    }
    
    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
