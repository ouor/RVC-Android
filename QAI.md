# Qualcomm AI Hub 운용 노트

> ContentVec/RMVPE/RVC 합성기 ONNX를 Snapdragon 8 Elite (V79) 컨텍스트
> 바이너리로 컴파일하면서 만난 시행착오 모음. 같은 함정에 다시 빠지지
> 않기 위한 기록.

## 워크플로 요약

```
ONNX (FP32 or FP16, dynamic shapes)
    │
    │  hub.upload_model(path) → Model(model_id)
    ▼
[staticize] dynamic axes → static dim_value
    │
    │  Plan A: hub.submit_compile_job(model, options="--target_runtime onnx",
    │           input_specs={...})
    │  Plan B: 로컬에서 onnx.shape_inference / onnxsim
    ▼
static ONNX
    │
    │  hub.submit_quantize_job(model, calibration_data,
    │       weights_dtype=INT8, activations_dtype=INT16)
    ▼
QDQ ONNX (W8A16)
    │
    │  hub.submit_compile_job(model, device, input_specs,
    │       options="--target_runtime qnn_context_binary --truncate_64bit_io")
    ▼
.bin (V79 context binary)
```

각 단계마다 5~15분. 직렬로 30분 소요. 병렬 모델 작업 시 따로따로 잡 던질 수
있음.

## 핵심 API 호출 패턴

```python
import qai_hub as hub

# 디바이스 핀 (sm8750-ac, V79)
device = hub.Device(name="Samsung Galaxy S25")

# 업로드 (한 번 하면 model_id로 재사용 가능)
raw = hub.upload_model("path/to/model.onnx")
print(raw.model_id)
# 재사용:
raw = hub.get_model("mnwrxz9rm")

# 양자화 (W8A16)
quantize_job = hub.submit_quantize_job(
    model=raw,                                           # 또는 static 모델
    calibration_data={"input_name": [arr1, arr2, ...]},  # 입력 순서대로!
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT16,
    name="job-name",
)
qdq_model = quantize_job.get_target_model()  # 블로킹
if qdq_model is None:
    raise RuntimeError("quantize failed")

# 컴파일
compile_job = hub.submit_compile_job(
    model=qdq_model,
    device=device,
    input_specs={"input_name": ((1, 30720), "float32")},
    options="--target_runtime qnn_context_binary --truncate_64bit_io",
    name="job-name",
)
compiled = compile_job.get_target_model()
compiled.download("out.bin")

# 디버깅용 로그 다운로드
job = hub.get_job("jp1dl207p")
job.download_job_logs("logs/")  # 6 MB짜리 상세 빌드 로그
print(job.get_status().code, job.get_status().message)
```

## 함정 모음 (만난 순서대로)

### 1. `extract_model`이 value_info에 IO를 중복으로 남김

```
"Tensors {'audio', 'unit12'} occur in value_info but also in model IO"
```

ContentVec ONNX는 출력 셋이 (`units9`, `unit12`, `unit12s`) — `unit12`만
필요해서 `onnx.utils.extract_model`로 자르면 ONNX validator가 거부.

**해결**: 추출 후 IO 이름과 같은 `value_info` 항목을 명시적으로 제거.

```python
trimmed = onnx.load(dst_path)
io_names = {i.name for i in trimmed.graph.input} | {o.name for o in trimmed.graph.output}
keep = [vi for vi in trimmed.graph.value_info if vi.name not in io_names]
del trimmed.graph.value_info[:]
trimmed.graph.value_info.extend(keep)
onnx.save(trimmed, dst_path)
```

### 2. `submit_quantize_job`은 dynamic shape를 거부

```
You can convert dynamic shapes to static using a compile job that
provides the input shapes and targets ONNX runtime
(`--target_runtime onnx`).
quantize failed
```

**해결 옵션**:
- **(A) AI Hub 사이드 staticize**: `submit_compile_job(options="--target_runtime onnx", input_specs={...})`
- **(B) 로컬 staticize**: dim_param → dim_value 치환 + shape inference

ContentVec은 (A)로 잘 됨. RMVPE는 STFT 때문에 (A)가 망가뜨려서 (B) 필요.

### 3. RMVPE: AI Hub staticize가 STFT의 window input 망가뜨림

```
ONNXRuntimeError: Node (/rmvpe/mel_extractor/STFT) Op (STFT)
  [ShapeInferenceError] window input must have rank = 1
```

`--target_runtime onnx` 컴파일이 STFT 주변 텐서 shape를 잃어버림. 로컬에서
`onnx.shape_inference.infer_shapes`만 돌려도 STFT 다음 Reshape 출력이
`[1, 'unk__4']`로 남아 같은 에러.

**해결 (절반)**: `onnxsim.simplify(model, overwrite_input_shapes={...})`로
constant folding + 강한 shape 추론. 로컬 ORT 1.23은 통과.

**여전히 문제**: AI Hub 사이드 quantize ORT(추정 더 오래된 버전)는 그래도
"window must have rank = 1" 거부. `onnxsim`이 도움 안 됨. RMVPE는 W8A16
양자화 보류 — fp16 baseline 유지.

### 4. STFT 우회 가능성

같은 STFT 노드의 window initializer 자체는 `[1024]` rank 1 (확인됨).
local ORT 1.23은 모델 정상 로드. 결국 AI Hub 측 STFT validator가 다름.

**대안**:
- (i) STFT 노드를 Conv1D + DFT 매트릭스로 치환 (ONNX 수술, 작업량 큼)
- (ii) AI Hub support에 ORT 버전 업데이트 요청
- (iii) RMVPE는 fp16 유지하고 다른 두 모델만 양자화

이번에는 (iii)로 갔음.

### 5. Calibration data dict 키 순서가 중요

```
Calibration data set has input 'pitch' but expected 'feats'
```

AI Hub가 calibration_data를 ONNX input 순서대로 매칭한다. Python `set`으로
키를 만들면 순서가 달라져 첫 키부터 잘못 매칭.

```python
# ❌ 망가짐
required = {"feats", "p_len", "pitch", "pitchf", "sid"}  # set!
calibration_data = {name: [...] for name in required}

# ✅ ONNX input 순서대로 list
required = ["feats", "p_len", "pitch", "pitchf", "sid", "rand_noise"]
calibration_data = {name: [...] for name in required}
```

`dict`는 Python 3.7+에서 삽입 순서 유지하므로 list 순회로 만들면 OK.

### 6. `RandomUniformLike` op이 W8A16 컴파일에서 거부됨

```
Unsupported input/output datatypes requested for the HTP Op
'RandomUniformLike' in the node '/dec/m_source/l_sin_gen/RandomUniformLike'.
Supported I/O datatype sets for the configuration: OTHERS
QnnBackend_validateOpConfig failed 3110
```

NSF SineGen의 초기 phase offset에 쓰는 `torch.rand_like`가 ONNX
`RandomUniformLike`로 export됨. FP16 컴파일은 통과. 하지만 양자화 후 입력
텐서가 UFIXED가 되면 HTP가 거부.

**해결**: ONNX 수술로 `RandomUniformLike` → `Constant of zeros[1, 1]`로
교체. 이미 `tools/export_static_synthesizer.py`가 `torch.randn{,_like}` →
zeros로 무력화하는 정책과 일관됨. 음질 영향: NSF 초기 phase가 random이
아니라 0 — unvoiced 구간 감각에 약간 영향 가능, PoC에는 무해.

```python
def zero_out_random_ops(src_path, dst_path):
    model = onnx.load(src_path)
    for i, n in enumerate(list(model.graph.node)):
        if n.op_type != "RandomUniformLike":
            continue
        out_name = n.output[0]
        # output shape lookup from value_info → fallback (1,1)
        zero = onnx.numpy_helper.from_array(np.zeros(shape, np.float32),
                                            name=out_name + "_zero")
        new_node = onnx.helper.make_node("Constant", [], [out_name],
                                          value=zero, name=n.name+"_zero")
        model.graph.node.remove(n)
        model.graph.node.insert(i, new_node)
    onnx.save(model, dst_path)
```

### 7. 양자화 후 MatMul I/O dtype 불일치 (Synth)

`RandomUniformLike` 제거 후 다음 컴파일 에러:

```
Unsupported input/output datatypes requested for the HTP Op 'MatMul' in
the node 'node_MatMul_2'. There is a mismatch for the combination of
inputs and outputs.
Supported I/O datatype sets for the configuration: BF16, FP16, INT16, INT8
```

MatMul 노드의 입력 두 개 dtype이 HTP가 받는 4가지 조합(BF16/FP16/INT16/INT8
각각 동일) 어느 것도 매치 안 됨 — 한쪽은 양자화(UFIXED) 다른 한쪽은 다른
dtype.

원인 추정: 합성기의 INT64 입력(`pitch`/`p_len`/`sid`)은 임베딩 lookup
(`Gather`) 통해 텐서로 변환된 후 MatMul로 들어간다. 양자화 후
`emb_g(sid).unsqueeze(-1)` 같은 패턴에서 한 분기는 양자화되고 한 분기는
INT32(truncate된 sid)로 남아 MatMul에서 충돌.

**해결 못 함 (이번 단계)**:
- AI Hub Python API에 per-op 양자화 skip list 미노출
- QDQ ONNX를 받아서 직접 손보는 게 가능하긴 하나 어떤 MatMul이 깨지는지
  매핑이 어려움 (양자화 단계가 노드 이름을 `node_MatMul_*`로 일률 변경)
- W8A8로 시도 (모든 dtype 일치) — 별개 실험으로 미뤄둠

**Synth 양자화 결론**: 이번 단계에서는 fp16 baseline 유지. 추후 AI Hub의
다음 SDK 업데이트 또는 per-op skip 옵션이 추가되길 기다리거나, QDQ ONNX
post-process로 문제 MatMul만 dequant 처리하는 별도 작업.

### 8. Calibration corpus 빌드: 입력별 리샘플링 일치

세 모델 다 16kHz fp32 audio 받음. 보정 corpus도 16kHz로 정규화. 다양한
SR(22050/44100/48000)이 섞이면 ffmpeg로 통일.

`build_calibration_corpus.py`:
- `ffmpeg -i x.wav -ac 1 -ar 16000 -f f32le pipe:1` → numpy
- 1.92s (30720 samples) 윈도우, 50% overlap, RMS floor 0.005로 무음 제외
- 라운드-로빈으로 화자 균형
- HuBERT/RMVPE용 50 sample 추출

Synth 보정은 별도 (`build_synth_calibration.py`):
- audio_30720 50개 → ORT로 HuBERT 6×5120 + RMVPE 1×30720 실행
- HuBERT 결과 90 frames @ 50 Hz → upsample 2× → pad to 192
- RMVPE 결과 193 frames → truncate to 192
- melQuantize → pitch[1, 192] int64
- rand_noise는 seeded Gaussian
- 모두 ONNX dtype에 맞춰 (feats fp16, pitch/p_len/sid int64, pitchf fp32, rand_noise fp16)

### 9. `--truncate_64bit_io`는 합성기 + 컴파일 단계에서 모두 필요

합성기 입력 `p_len/pitch/sid`가 INT64. HTP는 INT64 IO 거부.
`--truncate_64bit_io`로 INT32 변환. 값 범위 안전:
- p_len = 192 (int32 범위)
- pitch ∈ [1, 255] (mel quant)
- sid 작은 정수

### 10. ContentVec VTCM 한계 (양자화로 해소되지 않음)

V79 VTCM = 8 MiB. ContentVec @ 30720은 conv_layers.1이 48 MiB 요구 → 컴파일
실패. 5120으로 낮춤.

```
Requires 0x2ff0000 bytes of TCM, which is greater than the TCM size of 0x800000
```

W8A16 양자화는 weight ÷2지만 activation은 INT16=2B 그대로 → VTCM 안 줄어듦.
**A8 (activation 8-bit) 시도해볼 가치 있음** — 별개 후속 실험.

### 11. 작업 시간 분포

(measured 2026-04-27)

| Step | 시간 |
|---|---|
| upload (350 MiB ONNX) | ~30s |
| upload calibration dataset (5~10 MiB) | ~2s |
| `submit_compile_job(--target_runtime onnx)` (staticize) | ~5min |
| `submit_quantize_job` (50 calib samples) | ~5-15min |
| `submit_compile_job(--target_runtime qnn_context_binary)` | ~10-15min |
| `download_job_logs` (6 MiB) | ~5s |

**총 직렬: ~30-45분/모델**. 두 모델 병렬로 돌리면 ~30-45분 wall clock.

## API 작은 실용 팁

- `model_id` / `dataset_id`로 재사용: 업로드 비싸므로 잡 실패해도
  업로드 데이터는 살림. `--model-id` flag를 모든 컴파일 스크립트에 추가해
  반복 실행 시간 단축.
- `quantize_job.get_target_model()`은 블로킹 — 실패 시 None 반환. 대시보드
  URL을 미리 출력해두면 디버깅 빠름.
- `job.download_job_logs(dir)`로 6 MiB 짜리 상세 텍스트 빌드 로그를
  내려받음. `[ ERROR ]` grep하면 진짜 원인 줄이 나옴.
- `job.url`은 그대로 `https://app.aihub.qualcomm.com/jobs/<job_id>/` 형태,
  브라우저에서 진행 상황 / 디바이스 프로파일 / build artifact 트리 다 볼 수
  있음.
- `hub.QuantizeDtype.{INT8,INT16,INT4}` 만 가능. fp16 보존하려면 양자화 자체를
  건너뛰어야 함 (per-op skip은 AI Hub Python API에서 직접 노출 안 됨).
- `hub.upload_dataset` 명시적으로 안 부르고 `submit_quantize_job`이 dict를
  받으면 자동으로 `Uploading dataset:` 단계 수행.
- compile/quantize options 문자열은 일종의 CLI flag 형태. 한 줄로 공백
  구분.

## 측정 결과 (2026-04-27)

### 컴파일 단계

| 모델 | FP16 .bin | W8A16 .bin | 비율 | 비고 |
|---|---|---|---|---|
| ContentVec (HuBERT) @ 5120 | 191 MiB | **99.3 MiB** | 52% | 컴파일 성공 |
| RMVPE @ 30720 | 181 MiB | (실패) | — | STFT shape inference 거부 (#3) |
| Synth (RVC v2) @ T=192 | 63.4 MiB | (실패) | — | MatMul I/O dtype 불일치 (#7) |

각 모델별 시도 횟수:
- HuBERT: 1차 (3-step AI Hub staticize 워크플로) 실패 → 2차 (model_id 재사용) 성공
- RMVPE: 3차 시도 모두 실패. AI Hub staticize → 로컬 staticize → onnxsim 강화 staticize
- Synth: 3차 시도 모두 실패. 입력 순서 → RandomUniformLike → MatMul

### 디바이스 dispatch 결과 (충격)

| 구성 | 청크당 시간 | 8청크 총 |
|---|---|---|
| HuBERT FP16 5120 baseline | ~28 ms × 6 = 168 ms | 8.9 s warm |
| HuBERT **W8A16** 5120 | **17,392 ms × 6 = 104 s** | **~13분** |
| 비율 | **~620배 느림** | 86× |

W8A16 양자화 .bin이 디바이스에서 거의 사용 불가 — 17초/호출. ContentVec.bin
컴파일은 두 단계 모두 SUCCESS 떨어졌고, IO 스키마도 fp32로 정상이지만, 실제
dispatch가 ARM CPU로 partition 떨어진 것으로 추정. 양자화 후 어떤 op가 HTP
미지원 dtype 조합인 거. profile job 안 돌려서 컴파일 직후 못 잡음.

**교훈**: 양자화 .bin은 반드시 `submit_profile_job`으로 device latency 사전
검증할 것. 컴파일 SUCCESS ≠ NPU dispatch.

## 후속 과제

- **RMVPE STFT 우회**: STFT 노드를 일반 ops (Conv1D + DFT 매트릭스 곱)로
  치환하는 ONNX 수술. 또는 AI Hub support 채널로 ORT 버전 이슈 보고.
- **Synth MatMul dtype 충돌**: int64 입력 + 임베딩 lookup + MatMul 패턴이
  HTP 양자화에서 어떤 노드가 깨지는지 QDQ ONNX 분석. 또는 W8A8 시도해
  모든 dtype 일치 강제.
- **HuBERT 30720 W8A8**: VTCM 해소 가능성 확인. activation 8-bit이 음성
  품질에 미치는 영향 청취 검증.
- **HuBERT W8A16 청취 검증**: 99 MiB 신규 .bin이 음성 변환 품질 유지하는지
  3-run 비교 — 현재 대기 중.
- **per-channel weight quantization**: 기본 per-tensor보다 정확도 ↑.
  AI Hub options에 노출되는지 확인.

## 중요 결론

세 모델 한꺼번에 W8A16 적용은 **파이프라인 별로 별도 우회 작업이 필요**
하다는 게 이번 라운드의 가장 큰 교훈. 모든 ONNX가 양자화 친화적이지 않으며,
op-level 호환성은 모델마다 다르다. local-dream의 SD1.5/SDXL이 깔끔히 W8A16
되는 건 그 그래프가 표준 conv/matmul/attention만 쓰기 때문 — RVC처럼 STFT,
Random*, Embedding lookup, NSF SineGen 같은 특수 노드가 섞이면 양자화 단계
별로 추가 작업이 필요하다.
