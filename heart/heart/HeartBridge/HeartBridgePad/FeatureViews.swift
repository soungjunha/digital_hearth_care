
import SwiftUI
import HealthKit
import Charts

// ---------------------------------------------------------
// [1] 대시보드
// ---------------------------------------------------------
struct DashboardView: View {
    // 뷰 구성을 위해 매니저는 필요하지만, 측정 버튼이 리스트로 이동했으므로
    // 여기서는 가이드 표시 용도로만 사용됩니다.
    @ObservedObject var manager: IOSHealthManager
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 30) {
                // 상단 타이틀
                VStack(alignment: .leading, spacing: 10) {
                    Text("안녕하세요, 회원님 👋").font(.title2).fontWeight(.bold).foregroundStyle(.secondary)
                    Text("하트브릿지\n사용 가이드").font(.system(size: 40, weight: .heavy)).foregroundStyle(.primary)
                    Text("iPad와 Watch를 활용한 정밀 분석").font(.title3).foregroundStyle(.secondary)
                }.padding(.top, 40)
                
                Divider()
                
                // ⭐️ [수정] 동시 측정 방법 가이드 (버튼 대신 가이드 배치)
                VStack(alignment: .leading, spacing: 15) {
                    Label("동시 측정 방법", systemImage: "info.circle.fill")
                        .font(.title2).fontWeight(.bold).foregroundStyle(.blue)
                    
                    // 가이드 카드
                    VStack(spacing: 20) {
                        HStack(spacing: 30) {
                            // iPad 아이콘
                            VStack {
                                Image(systemName: "ipad.landscape")
                                    .font(.system(size: 50))
                                    .foregroundStyle(.blue)
                                Text("iPad").font(.caption).bold()
                            }
                            
                            Image(systemName: "plus")
                                .font(.title).foregroundStyle(.gray)
                            
                            // Watch 아이콘
                            VStack {
                                Image(systemName: "applewatch.side.right")
                                    .font(.system(size: 50))
                                    .foregroundStyle(.red)
                                Text("Watch").font(.caption).bold()
                            }
                        }
                        .padding(.vertical, 10)
                        
                        VStack(alignment: .leading, spacing: 12) {
                            HStack(alignment: .top) {
                                Text("1.").bold().foregroundStyle(.blue)
                                Text("측정 기록 탭의 '동시 측정 시작' 버튼을 누르세요.")
                            }
                            HStack(alignment: .top) {
                                Text("2.").bold().foregroundStyle(.blue)
                                Text("오른손 검지를 iPad 카메라에 대세요.")
                            }
                            HStack(alignment: .top) {
                                Text("3.").bold().foregroundStyle(.blue)
                                Text("동시에 왼손 검지를 Watch 디지털 크라운에 대세요.")
                            }
                        }
                        .font(.subheadline)
                        .foregroundStyle(.primary.opacity(0.8))
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(20)
                    .background(Color.blue.opacity(0.05))
                    .cornerRadius(20)
                    .overlay(RoundedRectangle(cornerRadius: 20).stroke(Color.blue.opacity(0.2), lineWidth: 1))
                }
                
                Divider()
                
                // 가이드 그리드 (기존 유지)
                VStack(alignment: .leading, spacing: 20) {
                    HStack {
                        Image(systemName: "book.fill").font(.title).foregroundStyle(.blue)
                        Text("단계별 가이드").font(.title2).fontWeight(.bold)
                    }
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 20) {
                        GuideStepCard(step: 1, icon: "list.bullet.clipboard", title: "메뉴 이동", desc: "좌측 사이드바에서 '측정 기록' 메뉴를 선택하세요.")
                        GuideStepCard(step: 2, icon: "sensor.tag.radiowaves.forward", title: "측정 시작", desc: "상단의 '동시 측정 시작하기' 버튼을 누르세요.")
                        GuideStepCard(step: 3, icon: "waveform.path.ecg", title: "30초 측정", desc: "양손을 사용하여 30초간 움직임을 최소화하세요.")
                        GuideStepCard(step: 4, icon: "doc.text.magnifyingglass", title: "AI 분석", desc: "측정이 완료되면 자동으로 AI 리포트가 생성됩니다.")
                    }
                }
                Spacer(minLength: 50)
            }.padding(.horizontal, 40).padding(.bottom, 50)
        }
        .navigationTitle("홈")
    }
}

// ---------------------------------------------------------
// [2] 리스트 (수정: 삭제 및 편집 기능 추가)
// ---------------------------------------------------------
struct ECGHistoryList: View {
    @ObservedObject var manager: IOSHealthManager
    @Binding var selectedSample: ECGModel?
    @Environment(\.openURL) var openURL
    
    // 측정 화면 표시를 위한 State
    @State private var showLiveMeasurement = false
    
    var body: some View {
        List(selection: $selectedSample) {
            // 1. 동시 측정 버튼 섹션
            Section {
                Button(action: {
                    showLiveMeasurement = true
                }) {
                    HStack {
                        ZStack {
                            Circle().fill(Color.red.opacity(0.1)).frame(width: 40, height: 40)
                            Image(systemName: "heart.text.square.fill")
                                .font(.title2).foregroundStyle(.red)
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            Text("동시 측정 시작하기")
                                .font(.headline).foregroundStyle(.primary)
                            Text("iPad 카메라 + Watch 심전도")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.caption).foregroundStyle(.gray)
                    }
                    .padding(.vertical, 8)
                }
                .listRowBackground(Color(.systemBackground))
            }
            
            // 2. 개발자 도구 섹션
            Section("개발자 테스트 도구") {
                HStack(spacing: 12) {
                    Button(action: { manager.generateSingleNormal() }) {
                        HStack { Image(systemName: "checkmark.circle.fill"); Text("정상 생성").font(.caption.bold()) }
                            .foregroundStyle(.white).frame(maxWidth: .infinity).padding(12).background(Color.green).cornerRadius(10)
                    }.buttonStyle(PlainButtonStyle())
                    
                    Button(action: { manager.generateSingleAbnormal() }) {
                        HStack { Image(systemName: "exclamationmark.triangle.fill"); Text("비정상 생성").font(.caption.bold()) }
                            .foregroundStyle(.white).frame(maxWidth: .infinity).padding(12).background(Color.red).cornerRadius(10)
                    }.buttonStyle(PlainButtonStyle())
                }
                .listRowSeparator(.hidden)
                .listRowBackground(Color.clear)
            }
            
            // 3. 측정 기록 섹션 (⭐️ 수정됨)
            Section("측정 기록") {
                if manager.ecgSamples.isEmpty {
                    ContentUnavailableView("기록 없음", systemImage: "heart.slash", description: Text("데이터가 없습니다."))
                } else {
                    // ⭐️ ForEach로 감싸야 삭제/이동 기능이 작동합니다.
                    ForEach(manager.ecgSamples, id: \.self) { sample in
                        ECGRowView(sample: sample, isSelected: selectedSample == sample)
                            .tag(sample)
                            .listRowInsets(EdgeInsets(top: 6, leading: 10, bottom: 6, trailing: 10))
                            .listRowSeparator(.hidden)
                            .listRowBackground(Color.clear)
                    }
                    // ⭐️ [삭제 기능] 스와이프 삭제 및 편집 모드 삭제 활성화
                    .onDelete { indexSet in
                        manager.deleteRecord(at: indexSet)
                    }
                    // ⭐️ [이동 기능] 순서 변경 활성화 (필요 시 주석 해제)
                    // .onMove { indices, newOffset in
                    //    manager.moveRecord(from: indices, to: newOffset)
                    // }
                }
            }
        }
        .navigationTitle("측정 기록")
        .listStyle(.insetGrouped)
        .refreshable { manager.fetchECGHistory() }
        // ⭐️ [편집 버튼] 네비게이션 바 상단에 '편집' 버튼 추가
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                EditButton()
            }
        }
        .sheet(isPresented: $showLiveMeasurement) {
            LiveMeasurementView(healthManager: manager)
        }
    }
}
