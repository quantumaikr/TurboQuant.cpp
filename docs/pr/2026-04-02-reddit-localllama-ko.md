# TurboQuant.cpp — 1-bit KV 캐시, 품질 손실 제로. 35B MoE 검증.

TurboQuant 논문(ICLR 2026)을 순수 C로 구현한 독립 추론 엔진입니다. llama.cpp 포크가 아닌 자체 구축.

**하는 일:** Randomized Hadamard Transform + 부호 해싱으로 KV 캐시 key를 1비트로 압축합니다. 출력이 비압축 baseline과 바이트 동일합니다.

**검증된 결과:**

```
Qwen3.5-35B-A3B MoE (IQ2_XXS GGUF, 16GB Mac):
  baseline:   "The capital of France is Paris."
  1-bit KV:   "The capital of France is Paris."   ← 동일 출력

Gemma 3 4B (TQM, perplexity 101 토큰):
  FP16 KV:        PPL = 35.99
  1-bit K + Q4 V:  PPL = 36.00  (+0.03%)
```

1-bit attention 코사인 = 0.634 (정보이론 한계 2/pi와 일치). 비편향성: 10만 랜덤 벡터쌍에서 상대 bias < 0.2%.

**레포 내용:**

- 27K줄 C/Metal, 외부 의존성 없음
- GGUF 직접 로딩 (Q8_0, Q4_K_M, IQ2_XXS 검증)
- MoE 지원 (256 전문가, top-8, 공유 전문가)
- 1-bit 가중치 양자화 (8.4x 압축, 4B에서 zero quality loss)
- Metal GPU (Apple Silicon), CUDA/Vulkan/ROCm 컴파일 타겟
- 32개 테스트, ASan 클린
- Perplexity 측정, 활성값 프로파일링, 코드북 캘리브레이션

**정직한 한계:**

- 현재 CPU 추론 위주 (Metal MoE dispatch는 WIP)
- 35B: M3 16GB에서 ~1-4 tok/s (메모리 대역폭 한계)
- IQ2_XXS (2-bit 가중치)는 복잡한 추론에서 품질 제한 — KV 압축이 아닌 가중치 양자화의 한계
- Qwen3.5, Gemma 3에서만 테스트 (3개 아키텍처)

**알고리즘 (논문 기반):**

Key: 정규화 → RHT → Lloyd-Max 코드북 → QJL 부호 해시
1-bit: 부호만 → XOR + popcount attention

Value: 블록별 Q4 또는 Q2 양자화

논문은 일반 양자화기가 내적 추정에 체계적 편향을 도입함을 증명. RHT + QJL 보정이 이를 수학적으로 비편향으로 만듦.

https://github.com/quantumaikr/TurboQuant.cpp

논문: https://arxiv.org/abs/2504.19874
