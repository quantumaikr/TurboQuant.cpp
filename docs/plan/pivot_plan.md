# 방향 전환 실행 계획

> **날짜**: 2026-04-12
> **동기**: RLV 5-stage 파이프라인이 단순 vector-RAG를 못 이김. 핵심 강점에 집중.
> **DFlash 인사이트**: Apple Silicon은 bandwidth-bound — Metal 커널 최적화는 무의미, weight loading 최소화가 핵심

---

## 우선순위

| # | 작업 | 이유 | 예상 임팩트 |
|---|---|---|---|
| **P0** | unified 서버 속도 프로파일 + 최적화 | 3 tok/s → 목표 10+ tok/s | 사용자 체감 3배 |
| **P1** | KV 압축 실증 벤치마크 | 7× 압축 = 킬러 기능인데 데모가 없음 | 커뮤니티 설득력 |
| **P2** | RLV → 단순화 (RAG-lite) | 5-stage 복잡성 제거, 증명된 것만 남김 | 코드 유지보수성 |

---

## P0: unified 서버 속도 최적화

### 측정 (Karpathy R1)
```
현재: Phi-3.5-Q8_0, unified server, 8 threads → ~3 tok/s
목표: 같은 하드웨어에서 10+ tok/s (DFlash 기준 Phi-3.5는 6.5 tok/s 가능)
```

### 병목 후보
1. tokenizer 재로딩 (매 요청마다 `tq_load_tokenizer_from_gguf` 호출?)
2. KV state 재할당
3. thread pool 미활용 (매 요청마다 pthread 생성?)
4. 불필요한 메모리 복사

## P1: KV 압축 실증

### 측정
```
같은 모델, 같은 질문, FP32 KV vs turbo_kv_4b:
- 메모리 사용량 비교
- 응답 품질 비교 (PPL delta)
- 속도 비교
```

## P2: RLV 단순화

### 방향
- 5-stage → 2-stage (chunk + answer)
- locator의 BM25+RRF는 유지 (이건 좋음)
- select-by-index / verifier / researcher 제거
- 코드 1400줄 → 300줄
