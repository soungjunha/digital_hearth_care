import SwiftUI
import HealthKit
import Charts

struct DetailView: View {
    let sample: ECGModel
    let predictionManager = PredictionManager()
    
    @State private var ecgData: [Double] = []
    @State private var ppgData: [Double] = []
    @State private var aiScore: Double = 0.0
    @State private var isAnomaly: Bool = false
    @State private var message: String = ""
    @State private var isLoading: Bool = true
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // 헤더
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Label("AI Cardiac Care System", systemImage: "stethoscope").font(.caption2).fontWeight(.bold).foregroundStyle(.white).padding(.horizontal, 8).padding(.vertical, 4).background(Color.blue).cornerRadius(20)
                        Text("정밀 분석 리포트").font(.system(size: 34, weight: .bold))
                        Text(sample.date.formatted(date: .long, time: .shortened)).font(.title3).fontWeight(.semibold).foregroundStyle(.secondary)
                    }
                    Spacer()
                }.padding(.horizontal).padding(.top)
                
                if isLoading {
                    AnalyzingView()
                } else {
                    VStack(spacing: 20) {
                        DiagnosisCard(isAnomaly: isAnomaly, score: aiScore)
                        
                        // AI 소견 표시
                        HStack(alignment: .top, spacing: 16) {
                            Image(systemName: "quote.opening").foregroundStyle(.blue.opacity(0.5)).font(.title2)
                            VStack(alignment: .leading, spacing: 8) {
                                Text("AI 닥터의 소견").font(.footnote).fontWeight(.bold).foregroundStyle(.blue)
                                Text(message).font(.subheadline).lineSpacing(4).foregroundStyle(.primary.opacity(0.8))
                            }
                            Spacer()
                        }
                        .padding(20).background(Color.blue.opacity(0.03)).cornerRadius(20).overlay(RoundedRectangle(cornerRadius: 20).stroke(Color.blue.opacity(0.1), lineWidth: 1)).padding(.horizontal)
                        
                        GraphCard(title: "심전도 (ECG)", voltages: ecgData, color: .red)
                        
                        if !ppgData.isEmpty {
                            GraphCard(title: "맥파 (PPG)", voltages: ppgData, color: .green)
                        } else {
                             VStack(spacing: 10) {
                                Image(systemName: "sensor.tag.radiowaves.forward.slash").font(.largeTitle).foregroundColor(.gray)
                                Text("PPG 데이터 없음").font(.headline).foregroundColor(.secondary)
                                Text("워치 단독 측정 기록입니다.").font(.caption).foregroundColor(.gray)
                            }.frame(height: 100).frame(maxWidth: .infinity).background(Color.gray.opacity(0.05)).cornerRadius(16).padding(.horizontal)
                        }
                    }
                }
            }.padding(.bottom, 40)
        }.background(Color(.systemGroupedBackground))
        .onAppear { loadData() }
        .onChange(of: sample) { _, _ in loadData() }
    }
    
    func loadData() {
        isLoading = true; ecgData = []; ppgData = []
        
        if sample.isMock {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        ecgData = sample.ecgVoltages ?? []
                        ppgData = sample.ppgVoltages ?? []
                        
                        if let mockRes = sample.mockResult {
                            withAnimation(.easeInOut) {
                                isAnomaly = mockRes.isAnomaly
                                aiScore = mockRes.score
                                
                                // [수정] 삼항 연산자 뒤에 줄바꿈을 하여 문법 오류 해결
                                message = mockRes.isAnomaly
                                    ? "평소와 다른 불규칙한 심장 리듬 패턴이 감지되었습니다. 일시적인 스트레스나 카페인 섭취가 원인일 수 있으나, 증상이 지속된다면 전문의와의 상담을 권장합니다."
                                    : "현재 심장 리듬이 매우 안정적입니다. 정상 동 리듬(NSR) 범주 내에서 규칙적인 패턴을 보이고 있습니다."
                                
                                // [수정] 로딩 종료 코드를 별도 라인으로 분리
                                isLoading = false
                            }
                        }
                    }
                    return
                }
        
        if let cachedECG = sample.ecgVoltages, !cachedECG.isEmpty {
            self.ecgData = cachedECG
            self.ppgData = sample.ppgVoltages ?? []
            runPrediction()
        } else {
            guard let hkSample = sample.hkSample else { return }
            let store = HKHealthStore()
            var loadedECG: [Double] = []
            
            let query = HKElectrocardiogramQuery(hkSample) { (query, result) in
                switch result {
                case .measurement(let measurement):
                    if let voltage = measurement.quantity(for: .appleWatchSimilarToLeadI)?.doubleValue(for: .volt()) {
                        loadedECG.append(voltage)
                    }
                case .done:
                    DispatchQueue.main.async {
                        self.ecgData = loadedECG
                        self.ppgData = []
                        runPrediction()
                    }
                default: break
                }
            }
            store.execute(query)
        }
    }
    
    func runPrediction() {
        DispatchQueue.global(qos: .userInitiated).async {
            let result = predictionManager.predict(ecgRaw: self.ecgData, ppgRaw: self.ppgData)
            DispatchQueue.main.async {
                withAnimation(.easeInOut) {
                    self.isAnomaly = result.isAnomaly
                    self.aiScore = result.score
                    self.message = result.message
                    self.isLoading = false
                }
            }
        }
    }
}
