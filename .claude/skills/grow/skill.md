---
name: grow
description: "TurboQuant.cpp 지속성장 루프. 자동으로 현재 상태를 읽고, 가장 임팩트 있는 다음 작업을 선택하여 구현하고, 검증한다. 'grow', '성장', '계속', '다음', '진행', '개선' 요청 시 사용. 매 라운드마다 state.md를 읽고 업데이트하여 세션 간 연속성을 보장한다."
---

# Grow — Continuous Improvement Loop

매 라운드마다 자동으로: 상태 읽기 → 다음 작업 선택 → 구현 → 검증 → 상태 업데이트.

## Protocol

### Step 1: Read State
```
Read .claude/state.md → 현재 상태, 남은 과제, 우선순위 파악
```
이전 세션의 결과를 정확히 이어받는다. state.md가 없으면 score.sh와 WBS에서 상태를 재구성한다.

### Step 2: Select Next Task

"What Needs Work" 목록에서 **가장 임팩트 있는 항목** 선택:
- 사용자 직접 요청이 있으면 그것 우선
- 없으면: 버그 > 성능 > 기능 > 문서 순서

### Step 3: Implement

하나의 작업만 수행한다 (작고 정확하게):
- 코드 변경 전 관련 파일 읽기
- 변경 후 빌드 + 테스트 확인
- 테스트 실패 시 롤백

### Step 4: Verify

```bash
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

추가 검증 (해당 시):
- `./build/tq_run MODEL -t TOK -p "1+1=" -n 5` → "2" 확인
- `bash score.sh --quick`

### Step 5: Update State

`.claude/state.md` 업데이트:
- "What Works" 항목 추가
- "What Needs Work" 항목 제거 또는 순서 변경
- 새로 발견된 과제 추가
- Last updated 타임스탬프

### Step 6: Commit

```bash
git add -A && git commit -m "grow: [한줄 요약]" && git push
```

## Rules

- state.md는 **반드시** 매 라운드 끝에 업데이트
- 한 라운드에 **하나의 작업**만 (여러 작업 금지)
- 테스트 실패 시 **즉시 롤백** (score 하락 금지)
- 큰 변경은 에이전트에 위임 (직접 50줄 이상 코드 작성 금지)
