# RVC Android

<p align="center">
  <a href="../README.md">English</a> | <b>한국어</b>
</p>

<p align="center">
  <img src="img/logo.png" alt="RVC Android" width="440" />
</p>

<p align="center">
  손 안에서 끝나는 음성 변환 — <a href="https://github.com/w-okada/voice-changer">voice-changer</a> 호환 RVC
  모델로 목소리를 바꿔주는 안드로이드 앱.
</p>

> 비공식 커뮤니티 도구입니다.
> [voice-changer](https://github.com/w-okada/voice-changer) 프로젝트와는 무관합니다.

<p align="center">
  <img src="img/app.jpg" alt="모델과 입력 선택" width="240" />
  &nbsp;
  <img src="img/convert.jpg" alt="피치 설정 후 변환" width="240" />
  &nbsp;
  <img src="img/result.jpg" alt="결과 미리듣기 후 저장" width="240" />
</p>

## 이럴 때 좋아요

PC 없이, 네트워크 없이, 손에 든 폰 하나로 목소리를 바꿔보고 싶을 때가 있죠.

- **녹음한 목소리를 다른 화자로** — 짧은 클립을 마이크로 녹음하거나 파일로
  가져와서, 가지고 있는 RVC 모델의 목소리로 바꿔봅니다.
- **내 오디오를 밖으로 내보내고 싶지 않을 때** — 변환도 인코딩도 전부
  기기 안에서 끝납니다. 어떤 오디오도 서버로 올라가지 않으니 비행기 모드에서도
  동작합니다.
- **여러 모델을 그때그때 갈아끼우고 싶을 때** — 앱에 고정된 목소리가 없습니다.
  원하는 RVC ONNX 모델을 그 자리에서 골라 쓰면 됩니다.

## 무엇을 할 수 있나요

- **목소리를 바꿉니다** — 마이크 녹음 또는 오디오 파일(최대 60초)을 넣으면
  RVC 모델로 화자를 변환해줍니다.
- **원하는 포맷으로 내보냅니다** — WAV / MP3 / AAC / M4A / FLAC / OGG 중
  골라서 저장할 수 있어요.
- **모델은 직접 고릅니다** — Synth · HuBERT · RMVPE 세 모델을 파일 피커로
  선택합니다. 마이크로 바로 녹음하는 것도 됩니다.
- **변환 전에 미리 확인합니다** — 60초가 넘는 파일은 미리 걸러주고, 입력
  파형을 썸네일로 보여줍니다.
- **결과를 바로 들어봅니다** — 변환이 끝나면 모달 창이 떠서 그 자리에서 재생해
  볼 수 있고, "Save as…"로 내보냅니다.
- **다시 꺼내 볼 수 있어요** — 최근 변환 결과를 히스토리에 모아두니, 닫았던
  결과도 다시 열어 재생하거나 저장할 수 있습니다.
- **같은 모델이면 빠릅니다** — 한 번 불러온 모델은 메모리에 데워둔 채로 들고
  있어서, 같은 조합으로 다시 변환할 때 모델을 또 여는 시간을 건너뜁니다.

## 무엇이 필요한가요

- **디바이스**: arm64-v8a, Android 12 이상 (`minSdk 31`, `targetSdk 36`)
- **권한**: 마이크로 녹음할 때만 `RECORD_AUDIO` (파일 입력만 쓰면 필요 없어요)
- **모델 3종** — 모두 voice-changer 도구로 내보낸 ONNX여야 합니다. 세 슬롯
  모두 채워야 변환이 시작됩니다.

| 슬롯 | 모델 | 알아둘 점 |
|------|------|-----------|
| **Synth** | RVC synthesizer ONNX | voice-changer `export2onnx.py` 출력 형식이어야 해요. 모델 안에 `custom_metadata_props["metadata"]` JSON(`samplingRate` / `f0` / `embChannels` / `embedder` / `embOutputLayer` / `useFinalProj`)이 들어 있어야 합니다. 이게 없으면 "synth has no embedded metadata"로 거부돼요. |
| **HuBERT** | ContentVec / HuBERT 임베더 | voice-changer의 `content_vec_500.onnx`처럼 `unit12`(768d, v2), `units9`(256d, v1), `unit12s` 출력을 모두 노출하는 형태면 됩니다. 어떤 출력을 쓸지는 Synth 메타데이터를 보고 알아서 고릅니다. |
| **RMVPE** | 피치 추출기 | `waveform[1,N] f32`, `threshold[1] f32` 입력을 받는 형태. f0 모델일 때 필요합니다. |

Synth 모델은 [voice-changer](https://github.com/w-okada/voice-changer) 데스크톱
클라이언트에서 **export to onnx** 를 눌러 받습니다. 이 과정에서 필요한
메타데이터가 파일에 함께 박힙니다.

<p align="center">
  <img src="img/export.png" alt="voice-changer에서 ONNX 모델 내보내기" width="520" />
</p>

## 시작하기

직접 빌드해서 설치하는 경우:

```sh
./gradlew :app:assembleDebug
```

설치한 뒤에는:

1. 앱을 열고 **Synth · HuBERT · RMVPE** 모델 3종을 파일 피커로 고릅니다.
2. 오디오 파일을 가져오거나 **마이크로 직접 녹음**합니다 (최대 60초).
3. 필요하면 **피치(f0UpKey)** 와 **화자 ID(Speaker ID)** 를 조정합니다.
4. **Convert** 를 누르고, 결과가 뜨면 들어본 뒤 **Save as…** 로 저장합니다.

## 알아두면 좋아요

- **전부 기기 안에서, 오프라인으로** 동작합니다. 계정도 클라우드도 없고,
  오디오가 기기 밖으로 나가지 않습니다.
- 입력은 **60초까지**만 받습니다. 그보다 긴 파일은 변환 전에 걸러집니다.
- 모델이 클 수 있어서 `largeHeap` 으로 동작하며, 모델 파일은 캐시로 흘려보내
  mmap으로 읽습니다(자바 힙 OOM 회피).

## 내부 동작

변환 파이프라인 (`inference/RvcPipeline.kt` 기준):

1. 입력 디코드 → 16 kHz 모노 리샘플 (선형 보간)
2. HuBERT/ContentVec → `feats[1, T, C]` (50 fps)
3. RMVPE → `pitchf` + voice-changer와 비트 일치하는 mel `f0_coarse` 양자화,
   `f0UpKey` 반음 시프트 적용
4. 임베딩 50 fps → 100 fps 2× nearest 업샘플
   (PyTorch `F.interpolate(scale_factor=2)` 와 호환)
5. Synthesizer → 오디오, `[-1, 1]` 클립
6. ffmpeg-kit-audio(또는 in-process WAV)로 인코드 → SAF "Save as…"

모듈 구조:

```
app/src/main/java/com/ouor/rvcandroid/
├── MainActivity.kt
├── audio/         # 디코드/인코드, 리샘플, 녹음, 프리뷰 플레이어, 히스토리 LRU
├── inference/     # ORT 세션 캐시, HuBERT/RMVPE/Synth 래퍼, 메타데이터 파서
└── ui/            # Compose 화면 + ConversionViewModel (StateFlow 기반)
```

기술 스택:

- **Kotlin** + **Jetpack Compose / Material3** — UI
- **ONNX Runtime Android** — 추론
- **ffmpeg-kit-audio** (community fork, LGPL) — MP3 / AAC / M4A / FLAC / OGG 코덱
- **AndroidX Media3 ExoPlayer** — 변환 결과 프리뷰