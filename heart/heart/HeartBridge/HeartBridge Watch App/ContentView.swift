import SwiftUI
import WatchKit

// ---------------------------------------------------------
// 워치 앱: 단순화된 가이드 모드 (iPad 중심 설계)
// 역할: 사용자가 기본 '심전도' 앱을 실행하도록 유도
// ---------------------------------------------------------

struct ContentView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 15) {
                // 1. 헤더 영역
                Image(systemName: "heart.text.square.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(.red)
                    .symbolEffect(.pulse, options: .repeating, isActive: true)
                
                VStack(spacing: 4) {
                    Text("하트브릿지")
                        .font(.headline)
                        .fontWeight(.bold)
                    
                    Text("iPhone/iPad와 동시 측정")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                // 2. 측정 시작 버튼 (가이드 화면으로 이동)
                NavigationLink(destination: MeasurementGuideView()) {
                    HStack {
                        Image(systemName: "waveform.path.ecg")
                        Text("측정 가이드")
                            .fontWeight(.bold)
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .cornerRadius(20)
            }
            .padding()
        }
    }
}

// ---------------------------------------------------------
// 상세 가이드 화면
// ---------------------------------------------------------
struct MeasurementGuideView: View {
    // 화면 진입 시 햅틱 피드백
    @State private var animate = false
    
    var body: some View {
        ScrollView {
            VStack(spacing: 15) {
                Text("측정 준비")
                    .font(.headline)
                    .foregroundStyle(.green)
                
                Divider()
                
                // Step 1
                VStack(spacing: 5) {
                    Image(systemName: "digitalcrown.press.fill")
                        .font(.title2)
                        .foregroundStyle(.blue)
                    Text("1. 홈 버튼 누르기")
                        .font(.caption)
                        .fontWeight(.bold)
                    Text("현재 앱을 나가세요.")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                
                Image(systemName: "arrow.down")
                    .font(.caption2)
                    .foregroundStyle(.gray)
                
                // Step 2
                VStack(spacing: 5) {
                    Image(systemName: "waveform.path.ecg")
                        .font(.title2)
                        .foregroundStyle(.red)
                    Text("2. 심전도 앱 실행")
                        .font(.caption)
                        .fontWeight(.bold)
                    Text("빨간색 파형 아이콘")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                
                Image(systemName: "arrow.down")
                    .font(.caption2)
                    .foregroundStyle(.gray)
                
                // Step 3
                VStack(spacing: 5) {
                    Image(systemName: "timer")
                        .font(.title2)
                        .foregroundStyle(.orange)
                    Text("3. 30초 측정")
                        .font(.caption)
                        .fontWeight(.bold)
                    Text("iPad와 동시에 진행하세요.")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                
                Divider()
                    .padding(.top, 10)
                
                Text("측정이 끝나면 iPad에서\n자동으로 분석됩니다.")
                    .font(.system(size: 10))
                    .foregroundStyle(.gray)
                    .multilineTextAlignment(.center)
                    .padding(.bottom)
            }
        }
        .onAppear {
            WKInterfaceDevice.current().play(.click)
        }
    }
}

#Preview {
    ContentView()
}
