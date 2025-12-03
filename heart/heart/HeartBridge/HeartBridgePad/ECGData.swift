import Foundation
import HealthKit

// 1. 데이터 모델
struct ECGModel: Identifiable, Hashable {
    let id = UUID()
    let date: Date
    let duration: Double
    let averageHeartRate: Double
    let isMock: Bool
    
    let hkSample: HKElectrocardiogram?
    let ecgVoltages: [Double]?
    let ppgVoltages: [Double]?
    let mockResult: (isAnomaly: Bool, score: Double)?
    
    static func == (lhs: ECGModel, rhs: ECGModel) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

enum SideMenu: String, CaseIterable {
    case dashboard, history, profile, settings
}
