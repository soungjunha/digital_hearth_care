import Foundation
import HealthKit
import Combine

class HealthKitManager: ObservableObject {
    let healthStore = HKHealthStore()
    
    // UI에 표시할 심박수 데이터
    @Published var currentHeartRate: Int = 0
    
    private var heartRateQuery: HKQuery?
    
    // 1. 권한 요청 (심박수 읽기 권한만 있으면 됨)
    func requestAuthorization() {
        // 워치 앱은 가이드 및 BPM 표시용이므로 심박수 권한만 요청
        guard let heartRateType = HKObjectType.quantityType(forIdentifier: .heartRate) else { return }
        let typesToRead: Set<HKObjectType> = [heartRateType]
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            if success {
                print("⌚️ 워치 권한 허용됨")
                DispatchQueue.main.async { self.startHeartRateMonitoring() }
            } else {
                print("❌ 워치 권한 거부: \(String(describing: error))")
            }
        }
    }
    
    // 2. 실시간 심박수 모니터링 (대기 화면용)
    func startHeartRateMonitoring() {
        guard let sampleType = HKObjectType.quantityType(forIdentifier: .heartRate) else { return }
        
        // 최근 1시간 데이터부터 모니터링
        let predicate = HKQuery.predicateForSamples(withStart: Date().addingTimeInterval(-3600), end: nil, options: .strictEndDate)
        
        // 앵커 쿼리를 사용하여 실시간 업데이트 수신
        let query = HKAnchoredObjectQuery(type: sampleType, predicate: predicate, anchor: nil, limit: HKObjectQueryNoLimit) { (query, samples, deletedObjects, newAnchor, error) in
            self.updateHeartRate(samples: samples)
        }
        
        query.updateHandler = { (query, samples, deletedObjects, newAnchor, error) in
            self.updateHeartRate(samples: samples)
        }
        
        self.heartRateQuery = query
        healthStore.execute(query)
    }
    
    // 심박수 업데이트 헬퍼 함수
    private func updateHeartRate(samples: [HKSample]?) {
        guard let quantitySamples = samples as? [HKQuantitySample] else { return }
        
        // 가장 최근 샘플을 가져옴
        if let lastSample = quantitySamples.last {
            let bpm = Int(lastSample.quantity.doubleValue(for: HKUnit(from: "count/min")))
            DispatchQueue.main.async {
                self.currentHeartRate = bpm
            }
        }
    }
}
