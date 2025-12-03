
import SwiftUI
import Charts
import AVFoundation

struct LiveMeasurementView: View {
    @StateObject var cameraManager = PPGCameraManager()
    @ObservedObject var healthManager: IOSHealthManager
    @Environment(\.dismiss) var dismiss
    
    // ⏳ 상태 관리 변수들
    @State private var isStandby = true
    @State private var countdown = 10
    @State private var isPreparing = true
    @State private var showCompleteAlert = false
    @State private var timer: Timer?
    
    var body: some View {
        VStack(spacing: 30) {
            
            // [1] 상단 헤더 영역 (플래시 버튼 제거)
            ZStack(alignment: .topTrailing) {
                VStack(spacing: 10) {
                    Text(headerTitle).font(.title).bold().foregroundStyle(headerColor)
                        .animation(.easeInOut, value: isStandby)
                    Text(headerDescription).multilineTextAlignment(.center).foregroundStyle(.secondary)
                        .animation(.easeInOut, value: isStandby)
                }
                .frame(maxWidth: .infinity).padding(.top, 40)
                
                // ❌ 플래시 토글 버튼 영역 삭제
            }
            .padding(.horizontal)
            
            Spacer()
            
            // [2] 메인 콘텐츠 (대기 vs 카운트다운 vs 그래프)
            ZStack {
                if isStandby {
                    // 대기 상태 (버튼 대기)
                    VStack(spacing: 30) {
                        Image(systemName: "timer").font(.system(size: 80)).foregroundStyle(.blue).symbolEffect(.bounce, value: isStandby)
                        Button(action: { withAnimation { startCountdown() } }) {
                            Text("카운트다운 시작 (10초)").font(.title3.bold()).foregroundStyle(.white).frame(width: 250, height: 60).background(Color.blue).cornerRadius(30).shadow(radius: 5)
                        }
                    }.transition(.scale.combined(with: .opacity))
                } else if isPreparing {
                    // 카운트다운 (준비 중)
                    VStack(spacing: 20) {
                        ZStack {
                            Circle().stroke(lineWidth: 15).opacity(0.3).foregroundColor(.orange)
                            Circle().trim(from: 0.0, to: CGFloat(countdown) / 10.0)
                                .stroke(style: StrokeStyle(lineWidth: 15, lineCap: .round, lineJoin: .round)).foregroundColor(.orange)
                                .rotationEffect(Angle(degrees: 270.0)).animation(.linear(duration: 1.0), value: countdown)
                            Text("\(countdown)").font(.system(size: 80, weight: .bold)).contentTransition(.numericText())
                        }.frame(width: 200, height: 200)
                        Text("iPad와 Watch에\n손가락을 대고 유지하세요.")
                            .font(.headline).multilineTextAlignment(.center).foregroundStyle(.secondary)
                    }.transition(.scale.combined(with: .opacity))
                } else {
                    // 측정 중 (그래프)
                    VStack(spacing: 20) {
                        Chart {
                            ForEach(Array(cameraManager.ppgData.suffix(100).enumerated()), id: \.offset) { index, value in
                                LineMark(x: .value("Time", index), y: .value("Red", value)).foregroundStyle(.red).interpolationMethod(.catmullRom)
                            }
                        }.chartYAxis(.hidden).chartXAxis(.hidden).frame(height: 200).padding().background(Color.gray.opacity(0.1)).cornerRadius(16)
                        VStack(spacing: 8) {
                            ProgressView(value: cameraManager.progress).progressViewStyle(.linear).tint(.red)
                            HStack {
                                Text("\(Int(cameraManager.progress * 100))%").font(.caption).bold()
                                Spacer()
                                Text("남은 시간: \(Int(30 - (cameraManager.progress * 30)))초").font(.caption).monospacedDigit()
                            }
                        }.padding(.horizontal)
                    }.transition(.opacity)
                }
            }
            .padding()
            
            Spacer()
            
            // [3] 하단 취소 버튼
            Button(action: { stopAllActions(); dismiss() }) { Text("취소").font(.headline).foregroundStyle(.gray).padding() }
        }
        .onChange(of: cameraManager.progress) { _, newValue in
            if newValue >= 1.0 { finishMeasurement() }
        }
        .onDisappear { stopAllActions() }
        .alert("측정 완료", isPresented: $showCompleteAlert) {
            Button("확인") { healthManager.fetchLatestECGAndMerge(ppgData: cameraManager.ppgData); dismiss() }
        } message: {
            Text("데이터 수집이 완료되었습니다.\n잠시 후 '진단 기록' 화면에서\nAI 정밀 분석 결과를 확인해주세요.")
        }
    }
    
    // MARK: - Helpers
    var headerTitle: String {
        if isStandby { return "측정 대기" } else if isPreparing { return "준비하세요" } else { return "측정 중..." }
    }
    var headerColor: Color {
        if isStandby { return .blue } else if isPreparing { return .orange } else { return .red }
    }
    var headerDescription: String {
        if isStandby { return "아래 버튼을 누르면\n10초 카운트다운이 시작됩니다." } else if isPreparing { return "잠시 후 측정이 시작됩니다.\n준비 자세를 취해주세요." } else { return "손가락을 떼지 말고\n30초간 유지하세요." }
    }
    
    func startCountdown() {
        isStandby = false; isPreparing = true; countdown = 10
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            if countdown > 1 { countdown -= 1 } else { timer?.invalidate(); timer = nil; withAnimation { isPreparing = false }; cameraManager.startMeasurement() }
        }
    }
    func finishMeasurement() {
        guard !showCompleteAlert else { return }
        cameraManager.stopMeasurement()
        AudioServicesPlaySystemSound(1057)
        showCompleteAlert = true
    }
    func stopAllActions() {
        timer?.invalidate(); timer = nil
        cameraManager.stopMeasurement()
    }
}
