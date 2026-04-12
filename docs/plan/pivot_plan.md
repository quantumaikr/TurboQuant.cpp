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

### 병목 진단 + 수정 결과
1. ✅ **tokenizer 재로딩** — 매 요청마다 32K 토큰 파싱 → context 재사용으로 제거 (commit 6e39e64)
2. ⚠️ **KV state 이중 재할당** — server + quant_generate 둘 다 재생성 → server측만 수정, 내부 tq_generate는 별도 PR
3. 해당없음 (thread pool 이미 재사용)

**결과: ~2.0 tok/s → ~4.5 tok/s (warm, 2.3× 개선)**

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
