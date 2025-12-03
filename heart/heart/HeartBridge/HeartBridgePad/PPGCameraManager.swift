import AVFoundation
import UIKit
import Combine

class PPGCameraManager: NSObject, ObservableObject {
    private let captureSession = AVCaptureSession()
    private var videoOutput = AVCaptureVideoDataOutput()
    private var simulationTimer: Timer?
    
    @Published var currentPPGValue: Double = 0.0
    @Published var ppgData: [Double] = []
    @Published var progress: Double = 0.0
    @Published var isMeasuring = false
    
    // ❌ isFlashOn 변수 삭제
    
    private var startTime: Date?
    private let duration: TimeInterval = 30.0
    
    override init() {
        super.init()
        #if !targetEnvironment(simulator)
        DispatchQueue.global(qos: .userInitiated).async { self.checkPermissions() }
        #endif
    }
    
    func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: setupCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted { self.setupCamera() }
            }
        default: break
        }
    }
    
    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480
        
        // M1 아이패드 대응: 광각 카메라만 명시적으로 선택
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .back
        )
        
        guard let device = deviceDiscoverySession.devices.first else {
            captureSession.commitConfiguration(); return
        }
        
        // ❌ captureDevice 변수 저장 로직 제거
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) { captureSession.addInput(input) }
            
            try device.lockForConfiguration()
            
            // 60 FPS 설정 (안정성 보강)
            let targetFPS: Double = 60.0
            var bestFormat: AVCaptureDevice.Format?
            for format in device.formats {
                let ranges = format.videoSupportedFrameRateRanges
                let frameRates = ranges[0]
                if format.formatDescription.dimensions.width >= 640 && frameRates.maxFrameRate >= targetFPS {
                    bestFormat = format
                    break
                }
            }
            
            if let format = bestFormat {
                device.activeFormat = format
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: Int32(targetFPS))
                device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: Int32(targetFPS))
            }
            
            // Exposure/White Balance Lock (PPG 안정성 확보)
            if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
            if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
            device.unlockForConfiguration()
            
        } catch { print("Camera Config Error: \(error)") }
        
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
        
        captureSession.commitConfiguration()
    }
    
    func startMeasurement() {
        DispatchQueue.main.async {
            self.ppgData.removeAll()
            self.progress = 0.0
            self.isMeasuring = true
            // ❌ isFlashOn = true 로직 제거
        }
        self.startTime = Date()
        
        #if targetEnvironment(simulator)
        simulationTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [weak self] _ in
            self?.generateMockData()
        }
        #else
        DispatchQueue.global(qos: .userInitiated).async {
            // ⭐️ 플래시 켜는 로직 제거 ⭐️
            if !self.captureSession.isRunning { self.captureSession.startRunning() }
        }
        #endif
    }
    
    func stopMeasurement() {
        #if targetEnvironment(simulator)
        simulationTimer?.invalidate(); simulationTimer = nil
        #else
        DispatchQueue.global(qos: .userInitiated).async {
            // ⭐️ 플래시 끄는 로직 제거 ⭐️
            if self.captureSession.isRunning { self.captureSession.stopRunning() }
        }
        #endif
        DispatchQueue.main.async { self.isMeasuring = false }
    }
    
    // ❌ toggleFlash() 함수 제거
    // ❌ controlTorch() 함수 제거
    
    private func generateMockData() {
        guard let start = startTime else { return }
        let elapsed = Date().timeIntervalSince(start)
        let prog = elapsed / duration
        self.progress = min(prog, 1.0)
        if elapsed >= duration { stopMeasurement(); return }
        let fakeValue = sin(elapsed * 10) * 0.5 + 0.5 + Double.random(in: -0.05...0.05)
        self.currentPPGValue = fakeValue
        self.ppgData.append(fakeValue)
    }
}

extension PPGCameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // ... (이하 로직은 동일) ...
        guard isMeasuring, let start = startTime else { return }
        
        let elapsed = Date().timeIntervalSince(start)
        
        if elapsed >= duration {
            DispatchQueue.main.async { self.progress = 1.0; self.stopMeasurement() }
            return
        } else {
            if Int(elapsed * 60) % 5 == 0 {
                DispatchQueue.main.async { self.progress = elapsed / self.duration }
            }
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly); return
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var rSum: Int = 0; var count = 0; let step = 10
        
        let startY = height / 4; let endY = height * 3 / 4
        let startX = width / 4; let endX = width * 3 / 4
        
        for y in stride(from: startY, to: endY, by: step) {
            for x in stride(from: startX, to: endX, by: step) {
                let offset = y * bytesPerRow + x * 4
                rSum += Int(buffer[offset + 2])
                count += 1
            }
        }
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        
        if count > 0 {
            let avgRed = Double(rSum) / Double(count)
            DispatchQueue.main.async { self.currentPPGValue = avgRed; self.ppgData.append(avgRed) }
        }
    }
}
