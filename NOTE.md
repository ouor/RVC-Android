# RVC-Android — 개발 노트

> 오프라인 RVC (Retrieval-based Voice Conversion) 추론을 Android에서 NPU
> 가속으로 돌리기까지의 시행착오 기록. 최종 결과는 Snapdragon 8 Elite
> (SM-S931N) 위에서 15초 입력 오디오를 11.3초에 변환 (실시간보다 빠름).

## 최종 아키텍처

```
SAF picker (Compose UI)
        │
        ▼  WAV bytes
   wav read + linear resample (48 kHz → 16 kHz)
        │
        ▼  audio[1, T_in]
   ┌───────────────┐         ┌───────────────┐
   │ HuBERT (ORT)  │         │ RMVPE  (ORT)  │
   │  CPU EP       │         │  CPU EP       │
   │ unit12[768]   │         │ pitchf        │
   └───────┬───────┘         └───────┬───────┘
           │ feats[T_total, 768]     │ pitch[T_total]
           ▼                         ▼
    ┌──────────────────────────────────────┐
    │ static-T 청크 분할 (T=192 = 1.92s)   │
    └──────────┬───────────────────────────┘
               │  per chunk:
               │    feats[1, 192, 768] (fp16)
               │    rand_noise[1, 192, 192] (fp16, host-gen Gaussian)
               │    pitch/pitchf/p_len/sid
               ▼
       QnnSynthRunner (Kotlin, parent process)
               │  binary wire protocol over stdin/stdout
               ▼
    ┌────────────────────────────────────────────┐
    │ librvc_synth_runner.so (PIE executable)    │
    │   ProcessBuilder가 spawn → 별도 SELinux    │
    │   domain → unsigned PD 가능                │
    │   QNN 2.45 + V79 context binary (63.4 MiB) │
    │   audio[76800] (fp16)                      │
    └────────────────────────────────────────────┘
               │
               ▼  audio[T*400] fp32
        chunk concat + trim
               │
               ▼
        WAV write @ 40 kHz
```

## 시간순 마일스톤

### Phase 0 — PoC (ORT only, CPU)
- Compose 기반 SAF 파일 피커 + WAV r/w + 선형 리샘플러
- ORT 세션 래퍼 (env 1개, 모델별 session) — 처음엔 `readBytes()`로
  로드했다가 360 MiB ContentVec에서 OOM → `cacheDir`로 SAF URI를 stream
  복사 후 `createSession(filePath)`로 mmap.
- ContentVec ONNX 입출력은 voice-changer의 export와 동일 (`audio →
  units9/unit12/unit12s`) — 모델 metadata `embOutputLayer` +
  `useFinalProj`로 어떤 출력을 쓸지 선택.
- 합성기 입력은 fp16 — `OrtSession`에서 schema 보고 `android.util.Half`
  로 packing.
- 동작 검증: 15s 샘플 → 0B WAV 한 번 나왔다가, ContentVec 입력 이름
  맞추고 fp16 분기 추가 후 정상.

### Phase 1 — 청크 분할
- 30s+ 입력에서 ORT 세션이 토하던 시점부터 silent-point 분할 (RMS 기반)
  도입. 이후 정적 T로 통합되며 deprecated.

### Phase α — ORT 1.20 → 1.22.0 + QNN AAR variant
- `onnxruntime-android-qnn:1.22.0`이 Maven Central에 정식 publish되어
  있었음. AAR이 QNN SDK 2.25 호스트 라이브러리를 번들.
- `com.qualcomm.qti:qnn-runtime`(Maven Central)도 존재. 추후 SDK 2.45로
  swap하는 빌미가 됨.

### Phase β — NNAPI → QNN HTP EP
- 처음에 NNAPI EP를 시도. Snapdragon 8 Elite의 V79 코어를
  `ParseHtpArchitecture`가 거부 (`"79"` enum 부재) — NNAPI 길이 막힘.
- QNN HTP EP로 전환. `provider_options`에 `htp_arch=v79`, `backend_type=
  htp` 등 명시.

### Phase γ — 정적 T 합성기 감지
- 합성기 ONNX의 `feats` shape에서 dim 1이 정적이면 static-T 모드 진입.
- 비정적이면 그대로 dynamic, 정적이면 강제 청크 단위 추론.

### Phase δ — Static-T 파이프라인 청크 분할
- T=192 = 1.92s @ 100 fps (post-2× upsample). 청크 샘플 = 192 × 16 kHz
  / 100 = 30720. RMVPE/HuBERT는 한 번에 전체 입력 처리 후, 합성만 청크
  단위로 dispatch.
- 출력 길이를 원본 길이로 trim.

### Phase ε — QNN 파티션 통계
- ORT 1.22 + QNN EP에서 `Failed to finalize QNN graph.`로 거부. CPU fallback.
- Synthesizer/HuBERT/RMVPE 셋 다 CPU로 떨어짐 → ORT 1.22 + QNN 2.25는
  V79 합성기 그래프를 컴파일 못 함.
- QNN SDK 2.45를 직접 받아 `tools/setup_qnn_libs.sh`로 jniLibs/cpp/3rdparty에
  swap. AAR 안의 2.25와 충돌 → `packaging.jniLibs.pickFirsts`로 우리
  복사본 우선. 그래도 ORT 1.22 자체가 V79 partition을 못 잡음 → 벽.

### Phase ζ — ONNX → QNN context binary (AI Hub)
- Qualcomm AI Hub로 우회: `qai_hub.submit_compile_job` +
  `--target_runtime qnn_context_binary --truncate_64bit_io`.
- `tools/export_static_synthesizer.py`:
  - `torch.randn_like(m_p)` → 외부 입력 `rand_noise`로 패치
    (`types.MethodType`로 forward 교체)
  - global `torch.randn → torch.zeros` 오버라이드 (NSF SineGen 내부
    randn 무력화) → `RandomNormalLike` op 제거
- `tools/compile_synth_aihub.py` → `model_static_t192_qnn_qnn_v79.bin`
  (63.4 MiB). 입력 6개, 출력 1개 (`output_0`로 자동 rename됨).

### Phase η — Native C++ JNI for QNN runtime
- ORT 우회. `qnn_synth_runtime.{h,cpp}`로 dlopen 기반 QNN host 직접 작성:
  `QnnLog_create → QnnBackend_create → QnnDevice_create →
  QnnContext_createFromBinary → QnnGraph_retrieve`.
- V79는 unsigned PD를 요구 → `QnnHtpDevice_CustomConfig_t` +
  `useSignedProcessDomain=false`.
- `QnnDevice_create`에서 `loadRemoteSymbols failed with err 4000` —
  앱 프로세스의 SELinux 도메인이 fastrpc unsigned PD 접근 불가. JNI 길
  완전히 막힘.

### Phase θ — Kotlin QnnRvcSynthesizer + factory routing
- (η가 막혔지만 이미 작성한 layer는 살림)
- `RvcSynth` sealed interface로 ORT/QNN 양쪽 추상화.
- 모델 파일 확장자가 `.bin`이면 QNN 분기, `.onnx`면 ORT 분기.

### Phase κλμ — local-dream 아키텍처 도입 (최종 돌파)
- 참고 프로젝트: `C:/Users/hurwy/Codes/local-dream`이 같은 SELinux 벽을
  PIE 자식 프로세스로 우회.
- `librvc_synth_runner.so` — PIE 실행 파일을 jniLibs에 박아 ProcessBuilder로
  spawn. uid는 같지만 SELinux 도메인이 달라져서 unsigned PD 가능.
- AGP 9는 manifest에 `android:extractNativeLibs="true"`를 거부 → 대신
  `packaging.jniLibs.useLegacyPackaging = true`.
- 와이어 프로토콜: stdin/stdout LE 바이너리, magic
  `INFR/RESP/REDY/QUIT` + length-prefixed 배열.
  - `QnnSynthRunner.kt`가 부모 측. `DataInputStream/DataOutputStream`
    + 1 MiB `BufferedInput/Output`. stderr는 별도 thread로 logcat 흘림.
  - `rvc_synth_runner_main.cpp`가 자식 측. 한 번 init하고 무한 루프로
    INFR 받음. fp32→fp16 packing은 child에서 (호스트 페이로드 절감
    여지는 future work).
- 자식 프로세스는 `/vendor/lib64`를 dlopen 못 함 (default linker namespace의
  permitted_paths). 해결:
  - `<uses-native-library android:name="libcdsprpc.so" android:required="false"/>`
    → 시스템 lib 일부 노출 시도했지만 namespace 제한이 strict.
  - 결국 `tools/pull_vendor_deps.sh`로 NEEDED 재귀 walk → libcdsprpc.so,
    vendor.qti.hardware.dsp@1.0.so, vendor.qti.hardware.dsp-V1-ndk.so,
    libvmmem.so를 jniLibs/arm64-v8a에 번들.
  - `android.hardware.common-V2-ndk.so`는 `/system/lib64`에 있어 free.
- AI Hub가 `audio` 출력을 `output_0`으로 rename → 합성 직후
  `getOutputByIndex(0)`로 이름 의존 제거.

### 최종 검증 (Snapdragon 8 Elite, V79)
```
HuBERT (CPU, ORT)        1801 ms
RMVPE  (CPU, ORT)        1313 ms
Synth × 8 chunks (NPU)   ~485 ms × 8 = 3880 ms
total                    11321 ms (입력 15 s → 실시간보다 빠름)
```
- 모든 청크 `QnnGraph_execute done. status 0x0`.
- 출력 599892 samples @ 40 kHz, 원본 길이와 정확히 일치.

## 핵심 결정 vs. 폐기

| 결정 | 결과 |
|---|---|
| ContentVec 입력 schema = voice-changer 따라가기 | OK |
| ONNX I/O fp16 detection + Half packing | OK |
| ORT cacheDir mmap | 큰 모델 OOM 회피 |
| Silent-point 청크 분할 | 정적 T 도입과 함께 폐기 |
| NNAPI EP | V79 enum 미지원 → 폐기 |
| ORT QNN EP | V79 합성기 그래프 finalize 실패 → 폐기 |
| QNN SDK 2.45 swap (in-AAR 2.25 위에) | infrastructure로 잔존 |
| AI Hub 컴파일 (qnn_context_binary, fp16, --truncate_64bit_io) | 채택 |
| 외부 `rand_noise` 입력 + global randn 오버라이드 | 채택 |
| In-process JNI QNN host | unsigned PD 차단 → 폐기 |
| local-dream 스타일 PIE child + stdin/stdout | 채택 (최종) |
| 벤더 라이브러리 jniLibs 번들 (libcdsprpc.so + 의존) | 채택 (필수) |

## 주요 파일

```
app/src/main/cpp/
  CMakeLists.txt                 # PIE executable named librvc_synth_runner.so
  qnn_synth_runtime.{h,cpp}      # 최소 QNN host: init/setInput/execute/getOutput
  rvc_synth_runner_main.cpp      # main + INFR/RESP/REDY/QUIT 루프 + f32↔f16
  wire_protocol.h, wire_protocol_io.h  # LE 바이너리 프로토콜

app/src/main/java/com/ouor/rvcandroid/inference/
  RvcSynth.kt                    # sealed interface (ORT/QNN 공통)
  OrtRuntime.kt                  # ORT 세션 래퍼 + cacheDir mmap
  HubertExtractor.kt             # ContentVec content embed
  RmvpePitchExtractor.kt         # RMVPE pitch
  OrtRvcSynthesizer.kt           # ORT 합성기
  QnnSynthRunner.kt              # ProcessBuilder spawn + 와이어 프로토콜
  QnnRvcSynthesizer.kt           # QNN 합성기 (rand_noise 생성, pitch i64→i32)
  RvcPipeline.kt                 # 전체 파이프라인 + factory + static-T 청크 분할

tools/
  export_static_synthesizer.py   # voice-changer 합성기 → 정적 T ONNX
                                 # (forward 패치 + global randn 무력화)
  compile_synth_aihub.py         # AI Hub 컴파일 잡 → V79 context binary
  setup_qnn_libs.sh              # QNN SDK 2.45 → jniLibs/cpp/3rdparty
  pull_vendor_deps.sh            # /vendor/lib64 NEEDED 재귀 → jniLibs
```

## 향후 최적화 후보

1. **IPC 페이로드 절감**: 현재 청크당 ~735 KiB fp32 (feats + noise) 전송.
   호스트에서 fp16으로 packing해 절반으로 줄이면 청크당 IPC 비용 감소.
2. **공유 메모리**: `MemoryFile` / `ASharedMemory` 또는 mmap-shared 버퍼로
   stdin 복사 자체 제거.
3. **noise 자식 측 생성**: 호스트에서 Gaussian PRNG 돌리지 말고 자식이
   seed 받아 자체 생성하면 IPC -1/3.
4. **HuBERT/RMVPE도 NPU**: 동일 AI Hub 경로로 컴파일 시도. 단 ContentVec은
   361 MiB / RMVPE는 345 MiB라 컨텍스트 바이너리 사이즈 검토 필요.
5. **세션 prewarm**: 첫 청크가 두 번째보다 약간 느림 (cold path) — UI에서
   파일 로드 시점에 미리 INFR 한 번 던져 두는 식.
