
import Foundation
import CoreML

class PredictionManager {
    // 모델 로드
    let model: HeartAnomalyDetector?
    
    // 임계값 (모델 학습 시 결정된 재구성 오차 임계값)
    // 0.02는 예시이며, 실제 정상 데이터 테스트 후 조정 필요
    let threshold: Double = 0.02
    
    init() {
        do {
            let config = MLModelConfiguration()
            self.model = try HeartAnomalyDetector(configuration: config)
        } catch {
            print("❌ 모델 로드 실패: \(error)")
            self.model = nil
        }
    }
    
    // ▶️ [핵심] 예측 함수 (ECG, PPG 동시 입력)
    func predict(ecgRaw: [Double], ppgRaw: [Double]) -> (isAnomaly: Bool, score: Double, message: String) {
        guard let model = model else { return (false, 0.0, "모델 오류") }
        
        // 데이터가 아예 없으면 분석 불가
        if ecgRaw.isEmpty { return (false, 0.0, "ECG 데이터 없음") }
        
        // 1. 전처리 (Resampling + Normalization)
        // 모델 Input Shape인 2560에 맞춰 리샘플링
        let processedECG = preprocessSignal(data: ecgRaw, targetLength: 2560)
        
        // PPG가 없는 경우(과거 데이터 등) 0으로 채움
        let processedPPG = ppgRaw.isEmpty
            ? Array(repeating: 0.0, count: 2560)
            : preprocessSignal(data: ppgRaw, targetLength: 2560)
        
        // 2. MLMultiArray 변환 (Shape: [1, 2, 2560])
        guard let inputArray = try? MLMultiArray(shape: [1, 2, 2560], dataType: .float32) else {
            return (false, 0.0, "텐서 변환 실패")
        }
        
        // 데이터 주입
        for i in 0..<2560 {
            // Channel 0: ECG
            inputArray[[0, 0, i] as [NSNumber]] = NSNumber(value: processedECG[i])
            // Channel 1: PPG
            inputArray[[0, 1, i] as [NSNumber]] = NSNumber(value: processedPPG[i])
        }
        
        // 3. 추론 및 결과 계산
        do {
            let output = try model.prediction(input_signal: inputArray)
            
            // MSE(Mean Squared Error) 계산 - 이상치 점수 산출
            // 재구성된 신호와 원본(전처리된) 신호의 차이 계산
            var mse: Double = 0.0
            
            for i in 0..<2560 {
                // ECG 채널 재구성 오차
                let orgECG = processedECG[i]
                let recECG = output.reconstructed[[0, 0, i] as [NSNumber]].doubleValue
                mse += pow(orgECG - recECG, 2)
                
                // PPG 채널 재구성 오차 (데이터가 있을 때만 반영)
                if !ppgRaw.isEmpty {
                    let orgPPG = processedPPG[i]
                    let recPPG = output.reconstructed[[0, 1, i] as [NSNumber]].doubleValue
                    mse += pow(orgPPG - recPPG, 2)
                }
            }
            
            // PPG가 있으면 2개 채널 평균, 없으면 ECG 채널만 평균
            let divider = ppgRaw.isEmpty ? Double(2560) : Double(2560 * 2)
            mse /= divider
            
            let isAnomaly = mse > threshold
            let msg = isAnomaly
                ? "⚠️ 이상 패턴 감지 (Score: \(String(format: "%.4f", mse)))"
                : "✅ 정상 리듬 (Score: \(String(format: "%.4f", mse)))"
                
            return (isAnomaly, mse, msg)
            
        } catch {
            print("❌ 추론 에러: \(error)")
            return (false, 0.0, "추론 실패")
        }
    }
    
    // ▶️ [유틸] 신호 전처리 함수 (Resampling -> Detrending -> MinMax Normalization)
    private func preprocessSignal(data: [Double], targetLength: Int) -> [Double] {
        guard !data.isEmpty else { return Array(repeating: 0.0, count: targetLength) }
        
        // A. 선형 보간 (Linear Interpolation)으로 길이 맞추기
        var resampled: [Double] = []
        let stride = Double(data.count - 1) / Double(targetLength - 1)
        
        for i in 0..<targetLength {
            let index = Double(i) * stride
            let leftIndex = Int(floor(index))
            let rightIndex = min(leftIndex + 1, data.count - 1)
            let weight = index - Double(leftIndex)
            
            let leftVal = data[leftIndex]
            let rightVal = data[rightIndex]
            let interpolated = leftVal * (1 - weight) + rightVal * weight
            resampled.append(interpolated)
        }
        
        // B. 디트렌딩 (평균 차감 - Baseline Wander 제거 효과)
        let mean = resampled.reduce(0, +) / Double(resampled.count)
        let centered = resampled.map { $0 - mean }
        
        // C. Min-Max 정규화 (0~1 범위)
        // 이상치가 있어도 0~1 안에 들어오게 하여 모델 폭주 방지
        let minVal = centered.min() ?? 0
        let maxVal = centered.max() ?? 1
        let denominator = maxVal - minVal
        
        if denominator == 0 { return centered.map { _ in 0.0 } }
        
        return centered.map { ($0 - minVal) / denominator }
    }
}
