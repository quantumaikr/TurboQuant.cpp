# TurboQuant.cpp v0.2 — 모든 주장에 숫자가 붙었습니다

V 캐시 양자화와 전수 검증 스위트를 출시합니다. 바뀐 것들을 정리합니다.

## v0.2에서 추가된 것

**V 양자화.** Key는 이미 1-bit였습니다. 이제 Value도 Q4 또는 Q2입니다.

```
Gemma 3 4B — 토큰당 총 K+V:

  FP16 기준:          136.00 KB
  1-bit K + Q4 V:      27.62 KB   (4.9x 압축)
  1-bit K + Q2 V:      19.12 KB   (7.1x 압축)
```

32K 컨텍스트에서 FP16 대비 3.7 GB 절약. "Paris"는 여전히 "Paris"입니다.

**검증.** NEON 버그를 발견하고 수정한 뒤, 전부 검증했습니다:

- 모든 NEON 경로를 스칼라 참조와 비교하는 14개 테스트
- Lloyd-Max 코드북 centroid가 이론값과 0.001 이내 일치 확인 5개 테스트
- Attention score 분포 보존 측정 8개 테스트
- 엣지케이스 29개 (NaN, Inf, 단일 토큰, 영차원, 만 개 키)
- ASan + UBSan 전체 26개 스위트 클린

## 핵심 수치

| 항목 | 측정값 | 재현 방법 |
|------|--------|----------|
| Attention 코사인 (1-bit) | 0.634 | `test_attention_distribution` |
| 이론 한계 (2/pi) | 0.637 | JL 문헌에서 증명 |
| 랜덤 K 코사인 | 0.089 | `test_attention_distribution` |
| 코드북 MSE vs 최적 | < 1.18배 | `test_codebook_theory` |
| RHT 오버헤드 | 147 ns/벡터 | `bench_kv_overhead` |
| 1-bit attention | 1.2 ns/키 | `bench_kv_overhead` |

1-bit 코사인 0.634는 2/pi = 0.637과 일치합니다. 이것은 결함이 아닙니다 — 부호 양자화의 정보이론적 최대값입니다. 우리 구현이 이론적 벽에 도달했습니다.

## 수정한 것

- **Q4 dequant NEON 버그**: nibble 인터리빙이 잘못되어 MSE가 300배 악화. 테스트로 발견, `vzip_u8`로 수정.
- **QJL sign bias**: `>= 0.0f` → `> 0.0f`, 11곳 (CPU/CUDA/Metal).
- **Norm overflow**: 큰 벡터에서 `sum += x*x` overflow 가능. max-abs rescaling 추가.
- **스레드 안전성**: 글로벌 워크스페이스 realloc에 mutex 보호.

## 정직한 부분

- 7.1x는 총 K+V입니다. 이전 "10.7x"는 K-only였고, 지금은 명확히 구분합니다.
- V 양자화(Q4/Q2) 시 출력이 기준에서 발산합니다. coherent하고 사실적으로 정확하지만, 바이트 동일은 아닙니다.
- 30/30 바이트 동일 결과는 K-only 모드(V는 FP16)에 해당합니다.
- 1-bit attention 코사인 = 0.634이지, 0.99가 아닙니다. 1비트의 최적값입니다. 더 높은 값이 필요하면 3-bit(0.918)를 쓰세요.

## 사용법

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build                          # 26/26 통과해야 합니다
./build/tq_run gemma3-4b.tqm -p "1+1=" -j 6 -n 5 -T 0.0 -k turbo_kv_1b -v q4 -M
```

---

[GitHub](https://github.com/quantumaikr/TurboQuant.cpp) | [릴리즈 노트](../RELEASE_NOTES.md) | [논문](https://arxiv.org/abs/2504.19874)
