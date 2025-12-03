import SwiftUI

struct ContentView: View {
    @StateObject var iosManager = IOSHealthManager()
    @State private var selectedMenu: SideMenu? = .dashboard
    @State private var selectedSample: ECGModel?
    @State private var columnVisibility = NavigationSplitViewVisibility.all

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            // [1열] 사이드바
            SidebarView(selectedMenu: $selectedMenu)
            
        } content: {
            // [2열] 목록/중간 콘텐츠
            if selectedMenu == .history {
                ECGHistoryList(manager: iosManager, selectedSample: $selectedSample)
            } else if selectedMenu == .dashboard {
                DashboardNoticeList() // 유의사항 리스트
            } else {
                ContentUnavailableView("준비 중", systemImage: "wrench.and.screwdriver")
            }
            
        } detail: {
            // [3열] 상세 화면
            if selectedMenu == .history {
                if let sample = selectedSample {
                    DetailView(sample: sample)
                } else {
                    EmptyStateView()
                }
            } else if selectedMenu == .dashboard {
                // ⭐️ [수정] manager를 전달하여 측정 버튼이 작동하도록 함
                DashboardView(manager: iosManager)
            } else {
                EmptyStateView()
            }
        }
        .navigationSplitViewStyle(.balanced)
        //.onAppear { iosManager.requestAuthorization() }
    }
}

// ---------------------------------------------------------
// 하위 컴포넌트들 (기존 코드 유지)
// ---------------------------------------------------------

struct SidebarView: View {
    @Binding var selectedMenu: SideMenu?
    var body: some View {
        List(selection: $selectedMenu) {
            Section {
                Label("홈 / 사용 가이드", systemImage: "house.fill").tag(SideMenu.dashboard)
                Label("측정 기록", systemImage: "list.bullet.clipboard").tag(SideMenu.history)
            } header: { Text("HeartBridge Care").font(.caption).fontWeight(.bold).foregroundStyle(.blue) }
            Section("설정") {
                Label("내 프로필", systemImage: "person.crop.circle").tag(SideMenu.profile)
                Label("설정", systemImage: "gearshape").tag(SideMenu.settings)
            }
        }
        .listStyle(.sidebar).navigationTitle("하트브릿지")
    }
}

struct DashboardNoticeList: View {
    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 10) {
                    Label("안전 유의 사항", systemImage: "exclamationmark.shield.fill")
                        .font(.headline).foregroundStyle(.orange).padding(.bottom, 5)
                    WarningRow(text: "본 앱은 '의료기기'가 아니며, 결과는 참고용입니다.")
                    WarningRow(text: "심장마비(Heart Attack) 등 응급 상황은 감지할 수 없습니다.")
                    WarningRow(text: "흉통 발생 시 즉시 119에 연락하십시오.")
                }
                .padding().background(Color.orange.opacity(0.05)).cornerRadius(12).listRowInsets(EdgeInsets())
            } header: { Text("Notice") }
        }.navigationTitle("홈").listStyle(.insetGrouped)
    }
}
