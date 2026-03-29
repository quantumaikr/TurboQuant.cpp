---
name: score
description: "TurboQuant.cpp의 5차원 스코어링 하네스를 실행하고 결과를 분석한다. 'score', '점수', '현재 상태', '스코어', '평가' 요청 시 사용. 구조/정확성/품질/성능/통합 5개 차원을 자동 측정하고, 병목 차원과 다음 최적 행동을 추천한다."
---

# Score — 5-Dimension Evaluation

TurboQuant.cpp 프로젝트의 완성도를 5개 차원으로 자동 측정한다.

## 실행

```bash
bash score.sh          # 전체 평가 (빌드+테스트+벤치마크)
bash score.sh --quick  # 빠른 평가 (빌드+테스트만)
```

## 5개 차원

| 차원 | 가중치 | 측정 항목 |
|------|--------|----------|
| **Structure** | 10 | 헤더/소스/테스트/스펙/WBS 존재 여부 |
| **Correctness** | 11 | CMake 빌드, 테스트 통과율, 경고 0, static_assert |
| **Quality** | 11 | 왕복 MSE < 0.01, attention cosine > 0.99 |
| **Performance** | 14 | 양자화/attention 처리량, 압축률, SIMD speedup |
| **Integration** | 6 | llama.cpp/vLLM/Python 플러그인, 예제, 문서 |

## 분석 프로토콜

1. score.sh 실행
2. `.score` 파일에서 수치 읽기
3. **가장 낮은 차원** 식별 → 이것이 다음 작업 대상
4. 해당 차원에서 **0점인 항목** 나열 → 구체적 행동 추천
5. `.score_history`가 있으면 **추세** 분석 (상승/정체/하락)
