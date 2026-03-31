# r/LocalLLaMA 한글 — 2026-04-01

## 제목

1-bit KV 캐시로 4-bit와 바이트 동일 출력 — 10.7배 압축, 30개 테스트 검증 (TurboQuant.cpp)

## 본문

KV 캐시를 **1비트**로 압축했는데 **4-bit uniform과 바이트 단위로 동일한 출력**이 나왔습니다. 비슷한 게 아니라, 토큰 하나하나 완전 동일합니다.

**Gemma 3 4B, greedy, 100 토큰:**

```
KV 타입        비트   압축률      출력
uniform_4b      4    3.8x      "Paris is the capital city of France."
turbo_kv_1b     1   10.7x      "Paris is the capital city of France."
                                ↑ 바이트 동일
```

수학, 코드, 지식, 한국어, 장문 등 30개 프롬프트 전량 일치.

**32K 컨텍스트 메모리:**

```
FP16 KV:          4,352 MB
TurboQuant 1비트:    408 MB   ← 3.9 GB 절약
```

**원리:** [TurboQuant 논문](https://arxiv.org/abs/2504.19874) (구글 리서치, ICLR 2026) 충실 구현. Random Hadamard Transform으로 채널 상관을 제거한 뒤 부호만 저장. Attention은 XOR + popcount 두 연산으로 수행.

핵심: MSE 최적 양자화기는 내적 추정에 편향을 만듭니다. 논문의 2단계 접근이 이 편향을 제거하고, 1비트에서도 내적 추정이 비편향(unbiased)입니다.

**재현:**
```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp
bash bench/kv_quality_bench.sh gemma3-4b.tqm
# → 30/30 byte-identical matches
```

순수 C, 의존성 없음, 1만줄. Gemma 3 (4B, 270M) + Qwen3.5 (0.8B) 지원.

GitHub: https://github.com/quantumaikr/TurboQuant.cpp
