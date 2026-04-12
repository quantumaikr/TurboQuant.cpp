# RLV Day 5 작업 계획서 — M3 환경 이관

> **작성일**: 2026-04-12
> **목적**: Phi-3.5-mini + unified 서버로 RLV wikitext 스트레스 테스트 실행
> **이관 사유**: 현재 머신에서 Phi-3.5 CPU 추론이 느려 (~1.15 tok/s) 벤치마크 진행 불가

---

## 1. 현재 상태 요약

### 완료된 것
- **RLV Day 3 PASS**: Acme 7/7 (Llama-3.2-3B, 184초) — `bench/rlv/eval/eval_acme.py`
- **RLV 5-stage 파이프라인**: 모든 코드 커밋 완료 (`91814d4`)
  - locator: non-LLM 키워드 점수 + 섹션 타이틀 보너스
  - lookup: select-by-index (구조화 문서) / direct-answer (내러티브)
  - verifier: question-grounding via locator 점수 재실행
  - gist: paragraph-aware chunker + narrative 대형 청크 (1500자)
- **Phi-3.5 unified 서버**: `tools/quant_server_unified.c` — quant.h 직접 사용, Phi-3.5 정상 작동 (`27671f5`)
- **Wikitext eval 하네스**: `bench/rlv/eval/eval_wikitext.py` — 10 질문, 3 시스템 비교

### 미완료 (M3에서 진행할 것)
1. `_llm.py` 수정 커밋 (unified 서버 연동)
2. Phi-3.5로 Acme 7문제 검증
3. Phi-3.5로 wikitext 10문제 스트레스 테스트 (RLV vs VR vs LC)
4. 결과 문서화 및 커밋

### 이전 wikitext 결과 (Llama-3.2-3B, 참고용)
| 시스템 | 정확도 | 비고 |
|---|---|---|
| RLV | 5/10 | locator 정확하지만 lookup에서 일부 실패 |
| long-context | 1/10 | cliff 11.6× 초과 → 모든 답 garbage |
| vector-RAG | 8/10 | 단순 키워드+직접 답변이 잘 작동 |

---

## 2. M3 환경 셋업

### 2.1 코드 최신화
```bash
cd ~/Dev/projects/quant.cpp  # 또는 적절한 경로
git pull origin main
```

### 2.2 uncommitted 변경 적용
`bench/rlv/stages/_llm.py`에 아래 변경이 필요합니다:

```python
# DEFAULT_MODEL과 DEFAULT_SERVER_BINARY를 아래로 변경:
DEFAULT_MODEL = REPO / "models" / "Phi-3.5-mini-instruct-Q4_K_M.gguf"
DEFAULT_SERVER_BINARY = REPO / "build_metal" / "quant-server-unified"
```

서버 시작 명령에서 `-k`/`-v`/`-H` 파라미터 제거 (unified 서버는 지원 안 함):
```python
# start_server() 내 cmd 빌드 부분:
is_unified = "unified" in str(binary)
if is_unified:
    cmd = [str(binary), str(model), "-p", str(port), "-j", str(threads)]
else:
    cmd = [str(binary), str(model), "-p", str(port), "-H", host,
           "-j", str(threads), "-k", kv_type, "-v", v_quant]
```

TQ_NO_METAL 환경변수 코드 제거 (더 이상 불필요):
```python
# 이 줄 삭제:
#     if "phi" in str(model).lower() or "Phi" in str(model):
#         env["TQ_NO_METAL"] = "1"
```

### 2.3 Phi-3.5 모델 다운로드
```bash
curl -L -o models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
# ~2.2GB
```

### 2.4 unified 서버 빌드
```bash
cc -O2 -o build_metal/quant-server-unified tools/quant_server_unified.c -lm -lpthread
```

### 2.5 빌드 검증
```bash
# 서버 기동 테스트
./build_metal/quant-server-unified models/Phi-3.5-mini-instruct-Q4_K_M.gguf -p 8421 -j 8 &
sleep 5

# 추론 테스트
curl -s http://127.0.0.1:8421/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10,"temperature":0}' | python3 -m json.tool

# 기대 출력: "content": "4" (또는 유사한 정확한 답변)
pkill -f quant-server-unified
```

**M3에서 예상 속도**: ~6.5 tok/s (PR #79 기준) → 질문당 ~10-15초

---

## 3. 실행 계획

### Step 1: Acme 7문제 검증 (Phi-3.5)
```bash
python3 bench/rlv/eval/eval_acme.py
```
**목표**: 7/7 PASS (Llama에서 이미 검증된 파이프라인이 Phi-3.5에서도 작동하는지)
**예상 시간**: ~2-3분 (7질문 × ~15초)

### Step 2: Wikitext 스트레스 테스트
```bash
# RLV + vector-RAG만 (long-context는 별도)
python3 bench/rlv/eval/eval_wikitext.py --systems rlv,vector-rag

# long-context 포함 전체 (느림 — LC는 12K 토큰 prefill 필요)
python3 bench/rlv/eval/eval_wikitext.py
```
**목표**: RLV > vector-RAG > long-context
**예상 시간**: RLV+VR만 ~10분, 전체 ~30분

### Step 3: 결과 분석 및 반복
- RLV < VR인 질문을 `--only N --verbose`로 디버깅:
  ```bash
  python3 bench/rlv/eval/eval_wikitext.py --only 3 --verbose --systems rlv
  ```
- locator가 잘못된 chunk를 고르면 → 키워드 가중치 조정
- lookup이 잘못된 문장 선택하면 → direct-answer 임계값 조정
- verifier가 잘못 판정하면 → question-grounding 임계값 조정

### Step 4: 결과 커밋
```bash
# _llm.py 변경사항 + 결과 문서 커밋
git add bench/rlv/stages/_llm.py docs/phase3_rlv_challenge.md
git commit -m "phase 3 day 5: Phi-3.5 unified server + wikitext stress test results"
git push origin main
```

---

## 4. 질문별 예상 동작

### Acme (5-section, ~500 tokens, sub-cliff)
| Q | 질문 | 예상 | Llama 결과 |
|---|---|---|---|
| 1 | Acme 총 매출? | PASS | PASS (847) |
| 2 | CTO 임명? | PASS | PASS (Santos) |
| 3 | 매출 성장 주요 동력? | PASS | PASS (Southeast Asia) |
| 4 | 동남아 전략 제안자? | PASS | PASS (James Park) |
| 5 | R&D 비율? | PASS | PASS (14%) |
| 6 | 전략 제안 이벤트? | PASS | PASS (Kyoto retreat) |
| 7 | 성장 지역 관련 리스크? | PASS | PASS (Currency) |

### Wikitext (3-article, ~12K tokens, 11.6× cliff)
| Q | 질문 | 핵심 답 | Llama RLV | Llama VR |
|---|---|---|---|---|
| 1 | Boulter 국적 | English | XX | OK |
| 2 | Herons 작가 | Stephens | OK | OK |
| 3 | Mercury Fur 감독 | Tiffany | XX | XX |
| 4 | Donkey Punch 감독 | Blackburn | XX | OK |
| 5 | Du Fu-Li Bai 만남 | 744 | OK | OK |
| 6 | An Lushan 반란 | 755 | OK | XX |
| 7 | Du Fu 강등 직위 | Commissioner | OK | XX |
| 8 | 과거 실패 이유 | dense/obscure | XX | OK |
| 9 | Kiss You 조회수 | 10.4M | XX | OK |
| 10 | Kiss You 감독 | Arnell | XX | XX |

**Phi-3.5에서 개선 기대**:
- Phi-3.5의 더 나은 instruction-following → select-by-index 정확도 ↑
- 더 나은 문장 이해력 → direct-answer 품질 ↑
- Q4 KV jitter 없음 (fp32 KV cache) → 깨끗한 출력

---

## 5. 핵심 파일 참조

| 파일 | 역할 |
|---|---|
| `bench/rlv/eval/eval_acme.py` | Acme 7문제 벤치마크 |
| `bench/rlv/eval/eval_wikitext.py` | Wikitext 10문제 스트레스 테스트 |
| `bench/rlv/stages/_llm.py` | 서버 연동 (수정 필요) |
| `bench/rlv/stages/locator.py` | 키워드 기반 chunk 선택 |
| `bench/rlv/stages/lookup.py` | select-by-index / direct-answer |
| `bench/rlv/stages/verifier.py` | question-grounding 검증 |
| `bench/rlv/stages/gist.py` | 문서 chunking |
| `bench/rlv/rlv_orchestrator.py` | 5-stage 오케스트레이터 |
| `tools/quant_server_unified.c` | Phi-3.5 unified 서버 |
| `docs/phase3_rlv_challenge.md` | 프로젝트 문서 (결과 기록용) |

---

## 6. 성공 기준

| 게이트 | 조건 | 상태 |
|---|---|---|
| D3 (Acme parity) | Phi-3.5 RLV ≥ 7/7 | 미검증 |
| D5 (wikitext breakthrough) | RLV > long-context AND RLV ≥ vector-RAG | 미검증 |
| D5 (cliff demonstration) | long-context < 3/10 (cliff 붕괴 증명) | ✅ 이미 확인 (1/10) |
