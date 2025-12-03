
import SwiftUI
import HealthKit
import Combine

class IOSHealthManager: ObservableObject {
    let healthStore = HKHealthStore()
    @Published var ecgSamples: [ECGModel] = []
    
    // IOSHealthManager 클래스 내부에 아래 함수를 추가하세요.

        // ⭐️ [New] 기록 삭제 함수
        func deleteRecord(at offsets: IndexSet) {
            // 배열에서 해당 인덱스의 데이터를 제거
            ecgSamples.remove(atOffsets: offsets)
            
            // (선택 사항) 만약 실제 HealthKit 데이터까지 지우고 싶다면
            // HKHealthStore.delete()를 호출해야 하지만,
            // 의료 데이터 안전을 위해 여기서는 앱 내 리스트에서만 제외합니다.
        }
        
        // ⭐️ [New] 기록 순서 변경 함수 (필요 시)
        func moveRecord(from source: IndexSet, to destination: Int) {
            ecgSamples.move(fromOffsets: source, toOffset: destination)
        }
    
    // 권한 요청
    func requestAuthorization() {
        let types: Set = [HKObjectType.electrocardiogramType()]
        healthStore.requestAuthorization(toShare: nil, read: types) { success, _ in
            if success { self.fetchECGHistory() }
        }
    }
    
    // 초기 데이터 로드 (과거 기록)
    func fetchECGHistory() {
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
        let query = HKSampleQuery(sampleType: HKObjectType.electrocardiogramType(), predicate: nil, limit: 30, sortDescriptors: [sort]) { query, samples, error in
            
            guard let samples = samples, let hkSamples = samples as? [HKElectrocardiogram] else { return }
            
            DispatchQueue.main.async {
                let realSamples = hkSamples.map { sample -> ECGModel in
                    let bpmUnit = HKUnit.count().unitDivided(by: .minute())
                    let heartRate = sample.averageHeartRate?.doubleValue(for: bpmUnit) ?? 0.0
                    let duration = sample.endDate.timeIntervalSince(sample.startDate)
                    
                    return ECGModel(
                        date: sample.startDate,
                        duration: duration,
                        averageHeartRate: heartRate,
                        isMock: false,
                        hkSample: sample,
                        ecgVoltages: nil, // 상세 화면에서 로드
                        ppgVoltages: nil, // 과거 기록은 PPG 없음
                        mockResult: nil
                    )
                }
                
                // 기존 테스트 데이터 유지하며 병합
                let existingMocks = self.ecgSamples.filter { $0.isMock }
                self.ecgSamples = (realSamples + existingMocks).sorted(by: { $0.date > $1.date })
            }
        }
        healthStore.execute(query)
    }
    
    // ⭐️ [핵심 추가] PPG 측정 직후 호출: 최신 ECG를 가져와서 합친 뒤 저장
    func fetchLatestECGAndMerge(ppgData: [Double]) {
        let ecgType = HKObjectType.electrocardiogramType()
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
        
        // 가장 최근 ECG 1개 조회
        let query = HKSampleQuery(sampleType: ecgType, predicate: nil, limit: 1, sortDescriptors: [sort]) { [weak self] query, samples, error in
            
            guard let self = self else { return }
            guard let latestECG = samples?.first as? HKElectrocardiogram else {
                print("❌ 최근 ECG 데이터를 찾을 수 없음")
                return
            }
            
            // 시간 동기화 체크 (5분 이상 차이나면 경고)
            let timeDiff = abs(latestECG.startDate.timeIntervalSince(Date()))
            if timeDiff > 300 {
                print("⚠️ 경고: ECG 데이터가 너무 오래되었습니다. (\(Int(timeDiff))초 전)")
            }

            // 전압 데이터 추출
            self.extractVoltages(from: latestECG) { ecgVoltages in
                DispatchQueue.main.async {
                    let bpmUnit = HKUnit.count().unitDivided(by: .minute())
                    let heartRate = latestECG.averageHeartRate?.doubleValue(for: bpmUnit) ?? 0.0
                    
                    // ECG + PPG 모델 생성
                    let newSample = ECGModel(
                        date: latestECG.startDate,
                        duration: latestECG.endDate.timeIntervalSince(latestECG.startDate),
                        averageHeartRate: heartRate,
                        isMock: false,
                        hkSample: latestECG,
                        ecgVoltages: ecgVoltages, // HealthKit ECG
                        ppgVoltages: ppgData,     // 방금 측정한 PPG
                        mockResult: nil
                    )
                    
                    // 리스트 최상단에 추가
                    self.ecgSamples.insert(newSample, at: 0)
                    print("✅ ECG(Watch) + PPG(iPad) 데이터 병합 완료")
                }
            }
        }
        healthStore.execute(query)
    }
    
    // 전압 추출 헬퍼 함수
    private func extractVoltages(from sample: HKElectrocardiogram, completion: @escaping ([Double]) -> Void) {
        var voltages: [Double] = []
        let query = HKElectrocardiogramQuery(sample) { (query, result) in
            switch result {
            case .measurement(let measurement):
                if let volt = measurement.quantity(for: .appleWatchSimilarToLeadI)?.doubleValue(for: .volt()) {
                    voltages.append(volt)
                }
            case .done:
                completion(voltages)
            default: break
            }
        }
        healthStore.execute(query)
    }
    
    // [TEST] 정상 데이터 생성
    func generateSingleNormal() {
        let now = Date()
        let normalECG = (0..<500).map { sin(Double($0) * 0.1) + Double.random(in: -0.05...0.05) }
        let normalPPG = (0..<500).map { sin(Double($0) * 0.1) * 0.5 + Double.random(in: -0.02...0.02) }
        
        let newSample = ECGModel(
            date: now,
            duration: 30.0,
            averageHeartRate: Double.random(in: 60...80),
            isMock: true,
            hkSample: nil,
            ecgVoltages: normalECG,
            ppgVoltages: normalPPG,
            mockResult: (isAnomaly: false, score: 0.002)
        )
        self.ecgSamples.insert(newSample, at: 0)
    }
    
    // [TEST] 비정상 데이터 생성
    func generateSingleAbnormal() {
        let now = Date()
        let abnormalECG = (0..<500).map { sin(Double($0) * 0.1) + Double.random(in: -0.8...0.8) }
        let abnormalPPG = (0..<500).map { sin(Double($0) * 0.1) * 0.5 + Double.random(in: -0.3...0.3) }
        
        let newSample = ECGModel(
            date: now,
            duration: 30.0,
            averageHeartRate: Double.random(in: 90...120),
            isMock: true,
            hkSample: nil,
            ecgVoltages: abnormalECG,
            ppgVoltages: abnormalPPG,
            mockResult: (isAnomaly: true, score: 0.12)
        )
        self.ecgSamples.insert(newSample, at: 0)
    }
}
