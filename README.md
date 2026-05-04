# RVC Android

Snapdragon 8 Elite 클래스 디바이스에서 **100% on-device**로 동작하는 RVC
(Retrieval-based Voice Conversion) 음성 변환 안드로이드 앱.

마이크 녹음 또는 오디오 파일(최대 60초)을 입력하면 voice-changer 호환 RVC
ONNX 모델로 화자를 변환해 WAV / MP3 / AAC / M4A / FLAC / OGG 중 원하는
포맷으로 내보낸다. 추론과 인코딩 모두 디바이스에서 끝나며 서버로 오디오를
전송하지 않는다.

## 요구사항

- **디바이스**: arm64-v8a, Android 12+ (`minSdk 31`, `targetSdk 36`)
- **권한**: 마이크 입력을 사용할 때만 `RECORD_AUDIO`
- **빌드**: JDK 11, Android Studio + Gradle wrapper 동봉
- **NPU 가속(선택)**: Qualcomm Hexagon V79 (Snapdragon 8 Elite).
  FP16 QNN context binary는 디바이스 dispatch가 검증됨. W8A16 양자화 빌드는
  컴파일은 성공하지만 디바이스에서 CPU fallback으로 떨어져 현재 실용 불가
  (자세한 내용은 `output/README.md`).

## 모델 (런타임에 SAF로 선택)

세 슬롯 모두 필수다.

| 슬롯 | 모델 | 비고 |
|------|------|------|
| **Synth** | RVC synthesizer ONNX | voice-changer `export2onnx.py` 출력 형식. `custom_metadata_props["metadata"]` JSON 필수 — `samplingRate` / `f0` / `embChannels` / `embedder` / `embOutputLayer` / `useFinalProj`. |
| **HuBERT** | ContentVec / HuBERT 임베더 | voice-changer의 `content_vec_500.onnx`처럼 `unit12` (768d, v2), `units9` (256d, v1), `unit12s` 출력을 모두 노출하는 형태. Synth 메타데이터에 따라 자동 선택. |
| **RMVPE** | 피치 추출기 | `waveform[1,N] f32`, `threshold[1] f32` 입력. f0 모델일 때 필수. |

## 파이프라인

`inference/RvcPipeline.kt` 기준:

1. 입력 디코드 → 16 kHz 모노 리샘플 (선형 보간)
2. HuBERT/ContentVec → `feats[1, T, C]` (50 fps)
3. RMVPE → `pitchf` + voice-changer와 비트 일치하는 mel `f0_coarse` 양자화,
   `f0UpKey` 반음 시프트 적용
4. 임베딩 50 fps → 100 fps 2× nearest 업샘플 (PyTorch
   `F.interpolate(scale_factor=2)` 와 호환)
5. Synthesizer → 오디오, `[-1, 1]` 클립
6. ffmpeg-kit-audio (또는 in-process WAV)로 인코드 → SAF "Save as…"

## UX

- 모델 3종 + 입력은 SAF 피커로 선택, 마이크로 직접 녹음도 가능
- 입력 파일은 60초 상한 사전 거부, 파형 썸네일 미리 표시
- 변환 결과는 모달 시트로 떠서 ExoPlayer로 즉시 프리뷰, "Save as…"로 내보내기
- 최근 변환 결과는 mtime-LRU 히스토리 카드에 보관, 다시 열 수 있음
- ORT 세션은 ViewModel이 warm 캐시로 들고 있어 같은 모델 조합으로 재변환할 때
  모델 재오픈 비용을 건너뜀

## 빌드

```sh
./gradlew :app:assembleDebug
```

NDK ABI는 `arm64-v8a`만 빌드된다 (`abiFilters` 고정 — ORT 네이티브 ~150 MB
절약). QNN 가속 경로는 vendored SDK가 필요하며 `qnn` 브랜치에서
`tools/pull_vendor_deps.sh`로 받아야 한다. `main` 브랜치는 ONNX Runtime 기본
실행 환경에서 동작한다.

## 모듈 구조

```
app/src/main/java/com/ouor/rvcandroid/
├── MainActivity.kt
├── audio/         # 디코드/인코드, 리샘플, 녹음, 프리뷰 플레이어, 히스토리 LRU
├── inference/     # ORT 세션 캐시, HuBERT/RMVPE/Synth 래퍼, 메타데이터 파서
└── ui/            # Compose 화면 + ConversionViewModel (StateFlow 기반)
```

`output/` — AI Hub 기반 ONNX → QNN context binary 변환 산출물 (gitignored,
약 2.1 GB). 재생성과 디바이스 검증 절차는 `output/README.md` 참고.

## 의존성

- ONNX Runtime Android — 추론
- ffmpeg-kit-audio (community fork, LGPL) — MP3 / AAC / M4A / FLAC / OGG
  코덱
- AndroidX Media3 ExoPlayer — 변환 결과 프리뷰
- Jetpack Compose + Material3 — UI
