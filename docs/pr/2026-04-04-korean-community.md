# 한국 커뮤니티 소개글 (2026-04-04)

대상: 커뮤니티 (GeekNews, AI Korea 등)

---

## 제목

quant.cpp — 33K LOC 순수 C로 만든 LLM 추론 엔진 (KV 캐시 4배 압축)

---

## 본문

로컬 LLM 추론을 위한 경량 C 엔진을 만들고 있습니다.

**quant.cpp** — https://github.com/quantumaikr/quant.cpp

핵심은 KV 캐시 압축입니다. 같은 하드웨어에서 context를 4배 늘릴 수 있습니다.

```
8GB 노트북 + Llama 8B:  16K → 61K tokens
16GB Mac + SmolLM2:      78K → 298K tokens
24GB 3090 + Llama 8B:    147K → 559K tokens
```

### 특징

- **Pure C, 33K LOC, 외부 의존성 0개** — llama.cpp(250K LOC)의 1/8 크기
- **Delta KV 압축** — 인접 key 차이만 저장 (비디오 P-frame 원리). 3-bit에서 PPL -3.2%
- **GGUF 호환** — llama.cpp용 모델 파일 그대로 사용
- **5개 아키텍처** — Llama, Gemma 3/4, Qwen3.5, Qwen-MoE

### llama.cpp와의 차이

llama.cpp를 대체하려는 게 아닙니다. 목적이 다릅니다.

llama.cpp는 모든 기능을 지원하는 프레임워크고, quant.cpp는 코드를 읽고 수정해서 내 프로젝트에 넣을 수 있는 라이브러리입니다. SQLite와 PostgreSQL의 관계와 비슷합니다.

KV 압축 성능 비교 (SmolLM2 1.7B 기준):
- llama.cpp Q4_0 KV: PPL +10.6%
- quant.cpp 4-bit K: PPL +0.0%

### Delta 압축이 뭔가요?

트랜스포머의 인접 key는 절대값 범위의 ~30%만 차이납니다. 이 차이(delta)만 저장하면 3-bit로도 품질 손실 없이 압축됩니다. 64 토큰마다 FP32 기준점(I-frame)을 두어 오차 누적을 방지합니다.

Delta 없이 3-bit → PPL +62%. Delta 적용 3-bit → PPL -3.2%.

### 정직하게

초기에 1-bit "무손실" 주장을 했었는데, 내부 FP32 fallback 버그로 인한 잘못된 측정이었습니다. 발견 후 모든 주장을 철회하고 코드를 수정했습니다. 현재 README의 모든 수치는 버그 수정 후 재측정한 값입니다.

### 빠른 시작

```bash
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
cmake -B build && cmake --build build -j$(nproc)
./build/quant model.gguf -p "hello" -k uniform_4b -v q4
```

피드백, 이슈, PR 환영합니다.

---

**QuantumAI** (https://quantumai.kr)
