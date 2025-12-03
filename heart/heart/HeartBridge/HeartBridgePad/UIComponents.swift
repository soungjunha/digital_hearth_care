import SwiftUI
import Charts

// 리스트 행
struct ECGRowView: View {
    let sample: ECGModel
    let isSelected: Bool
    
    var koreanDate: String {
        sample.date.formatted(.dateTime.locale(Locale(identifier: "ko_KR")).year().month().day().weekday())
    }
    var koreanTime: String {
        sample.date.formatted(.dateTime.locale(Locale(identifier: "ko_KR")).hour().minute())
    }
    
    var body: some View {
        HStack(spacing: 15) {
            ZStack {
                let isAnomaly = sample.isMock ? (sample.mockResult?.isAnomaly ?? false) : false
                Circle().fill(isAnomaly ? Color.red.opacity(isSelected ? 0.3 : 0.1) : Color.green.opacity(isSelected ? 0.3 : 0.1)).frame(width: 48, height: 48)
                Image(systemName: "waveform.path.ecg").foregroundStyle(isAnomaly ? .red : .green).font(.title3)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(koreanDate).font(.system(size: 16, weight: .bold)).foregroundStyle(Color.primary.opacity(isSelected ? 0.7 : 1.0))
                Text(koreanTime).font(.caption).foregroundStyle(Color.secondary)
            }
            Spacer()
            if sample.isMock {
                Text("TEST").font(.caption2.bold()).foregroundStyle(.white).padding(4).background(Color.gray.opacity(0.5)).cornerRadius(4)
            }
            HStack(spacing: 4) {
                Text("\(Int(sample.averageHeartRate))").font(.system(size: 18, weight: .heavy, design: .rounded)).foregroundStyle(Color.primary)
                Text("BPM").font(.caption2).foregroundStyle(Color.secondary)
            }
            .padding(.horizontal, 10).padding(.vertical, 6).background(Color(.secondarySystemBackground)).cornerRadius(8)
        }
        .padding(14).background(isSelected ? Color.blue.opacity(0.1) : Color(.systemBackground)).cornerRadius(16).shadow(color: .black.opacity(0.05), radius: 3, x: 0, y: 1)
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(isSelected ? Color.blue.opacity(0.5) : Color.gray.opacity(0.1), lineWidth: isSelected ? 2 : 1))
    }
}

// 그래프 카드
struct GraphCard: View {
    let title: String; let voltages: [Double]; let color: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Label(title, systemImage: "chart.xyaxis.line").font(.headline).foregroundStyle(.secondary)
            if !voltages.isEmpty {
                Chart { ForEach(Array(voltages.prefix(400).enumerated()), id: \.offset) { i, v in LineMark(x: .value("T", i), y: .value("V", v)).foregroundStyle(LinearGradient(colors: [color, color.opacity(0.3)], startPoint: .bottom, endPoint: .top)).interpolationMethod(.catmullRom).lineStyle(StrokeStyle(lineWidth: 2)) } }
                .chartYAxis { AxisMarks(position: .leading) }.chartXAxis(.hidden).frame(height: 250).padding().background(color.opacity(0.02)).cornerRadius(16)
            } else {
                VStack(spacing: 10) { Image(systemName: "lock.shield").font(.largeTitle).foregroundColor(.gray); Text("원시 데이터 접근 제한").font(.headline).foregroundColor(.secondary); Text("Apple 보안 정책상 PPG Raw Data는\n제공되지 않습니다.").font(.caption).foregroundColor(.gray).multilineTextAlignment(.center) }.frame(height: 250).frame(maxWidth: .infinity).background(Color.gray.opacity(0.05)).cornerRadius(16)
            }
        }.padding(25).background(Color(.systemBackground)).cornerRadius(24).shadow(color: .black.opacity(0.05), radius: 10, y: 5).padding(.horizontal)
    }
}

// 진단 카드
struct DiagnosisCard: View {
    let isAnomaly: Bool; let score: Double
    var body: some View {
        HStack(spacing: 0) {
            VStack { Image(systemName: isAnomaly ? "exclamationmark.triangle.fill" : "checkmark.seal.fill").font(.system(size: 50)).foregroundStyle(isAnomaly ? .red : .green).padding(.bottom, 5); Text(isAnomaly ? "주의 필요" : "정상 리듬").font(.title2).fontWeight(.bold).foregroundStyle(isAnomaly ? .red : .green) }.frame(width: 140)
            Divider().padding(.vertical)
            VStack(alignment: .leading, spacing: 8) {
                Text("AI 이상치 탐지 점수 (MSE)").font(.caption).fontWeight(.bold).foregroundStyle(.gray)
                HStack(alignment: .lastTextBaseline) { Text(String(format: "%.5f", score)).font(.system(size: 36, weight: .black, design: .monospaced)).foregroundStyle(.primary); Text("점").font(.body).foregroundStyle(.secondary) }
                GeometryReader { g in ZStack(alignment: .leading) { Capsule().fill(Color.gray.opacity(0.1)); Capsule().fill(LinearGradient(colors: isAnomaly ? [.orange, .red] : [.green, .mint], startPoint: .leading, endPoint: .trailing)).frame(width: min(g.size.width * CGFloat(score * 50), g.size.width)) } }.frame(height: 10)
            }.padding(.leading, 20); Spacer()
        }.padding(30).background(Color(.systemBackground)).cornerRadius(24).shadow(color: .black.opacity(0.05), radius: 10, y: 5).padding(.horizontal)
    }
}

// AI 코멘트
struct AICommentCard: View {
    let isAnomaly: Bool
    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: "quote.opening").foregroundStyle(.blue.opacity(0.5)).font(.title2)
            VStack(alignment: .leading, spacing: 8) {
                Text("AI 닥터의 소견").font(.footnote).fontWeight(.bold).foregroundStyle(.blue)
                Text(isAnomaly ? "평소와 다른 불규칙한 심장 리듬 패턴이 감지되었습니다. 전문의와의 상담을 권장합니다." : "현재 심장 리듬이 매우 안정적입니다. 정상 범주 내에서 규칙적인 패턴을 보이고 있습니다.")
                    .font(.subheadline).lineSpacing(4).foregroundStyle(.primary.opacity(0.8))
            }
            Spacer()
        }.padding(20).background(Color.blue.opacity(0.03)).cornerRadius(20).overlay(RoundedRectangle(cornerRadius: 20).stroke(Color.blue.opacity(0.1), lineWidth: 1)).padding(.horizontal)
    }
}

// 기타 부품들
struct AnalyzingView: View {
    var body: some View {
        VStack(spacing: 20) { ProgressView().scaleEffect(1.5); Text("AI가 정밀 분석하고 있습니다...").font(.subheadline).foregroundStyle(.secondary) }.frame(height: 300).frame(maxWidth: .infinity).background(Color(.systemBackground)).cornerRadius(20).padding(.horizontal)
    }
}
struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 24) { Image(systemName: "ipad.landscape").font(.system(size: 80)).foregroundStyle(.gray.opacity(0.3)); VStack(spacing: 8) { Text("기록을 선택하세요").font(.title2).fontWeight(.bold).foregroundStyle(.secondary); Text("좌측 목록에서 기록을 선택하면 AI 분석 리포트가 표시됩니다.").font(.body).foregroundStyle(.tertiary).multilineTextAlignment(.center) } }
    }
}
struct GuideStepCard: View {
    let step: Int; let icon: String; let title: String; let desc: String
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack { Image(systemName: icon).font(.title).foregroundStyle(.blue); Spacer(); Text("STEP \(step)").font(.caption).fontWeight(.bold).foregroundStyle(.gray.opacity(0.5)) }
            Text(title).font(.headline).fontWeight(.bold); Text(desc).font(.caption).foregroundStyle(.secondary).lineLimit(3).fixedSize(horizontal: false, vertical: true)
        }.padding().background(Color(.systemBackground)).cornerRadius(12).shadow(color: .black.opacity(0.05), radius: 3, x: 0, y: 2)
    }
}
struct WarningRow: View {
    let text: String
    var body: some View {
        HStack(alignment: .top) { Image(systemName: "info.circle.fill").foregroundStyle(.orange).padding(.top, 2); Text(text).font(.subheadline).foregroundStyle(.primary.opacity(0.8)) }
    }
}
