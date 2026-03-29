---
name: orchestrate
description: "TurboQuant.cpp 개발 오케스트레이터. 에이전트 팀을 구성하고, score 기반 Phase 전환, 병렬 작업 위임, Merge Gate를 자동 수행한다. '개발 시작', '하네스 실행', '팀 구성', '오케스트레이션' 요청 시 사용. 프로젝트의 핵심 개발 루프를 조율하는 상위 스킬."
---

# TurboQuant Orchestrator

TurboQuant.cpp 에이전트 팀을 조율하여 프로젝트를 자율적으로 발전시키는 통합 스킬.

## 아키텍처: 계층적 위임 + 팬아웃/팬인

```
[Orchestrator/Leader]
    ├── score.sh 실행 → Phase 결정
    ├── Phase에 맞는 에이전트 팀 구성
    │   ├── core-dev (알고리즘)
    │   ├── perf-dev (SIMD/GPU)
    │   └── qa (검증)
    ├── 워커들이 병렬 작업 (Fan-out)
    ├── Merge Gate: 하나씩 머지 + score 확인 (Fan-in)
    └── 다음 라운드 결정
```

## 워크플로우

### Step 1: 상태 평가
```bash
bash score.sh --quick
```
현재 점수, 가장 낮은 차원, 다음 WBS 항목을 파악한다.

### Step 2: Phase 결정 및 팀 구성

| Score | Phase | 전략 |
|-------|-------|------|
| < 0.05 | Foundation | 직접 수행 (단일) |
| 0.05~0.30 | Core | core-dev × 3 병렬 (polar, qjl, uniform) |
| 0.30~0.60 | Advanced | core-dev + perf-dev + qa 병렬 |
| 0.60~1.00 | Fine-tune | 단일 에이전트 정밀 작업 |

### Step 3: 작업 위임

에이전트 팀 모드:
```
TeamCreate("tq-round-N", members=[core-dev, perf-dev, qa])
TaskCreate("Implement tq_polar.c", owner="core-dev", blocked_by=[])
TaskCreate("NEON optimize polar", owner="perf-dev", blocked_by=["polar"])
TaskCreate("Verify polar boundaries", owner="qa", blocked_by=["polar"])
```

서브에이전트 모드 (ClawTeam):
```bash
clawteam spawn --team tq-N --agent-name core-polar --workspace \
  --task "Read .claude/agents/core-dev.md for your role. Implement tq_polar.c..."
```

### Step 4: Merge Gate

워커 완료 후, 하나씩 순차적으로 머지:
1. `pre_merge=$(git rev-parse HEAD)`
2. `git merge <worker-branch> --no-edit`
3. `bash score.sh --quick`
4. score 하락 → `git reset --hard $pre_merge` (롤백)
5. score 유지/상승 → 다음 워커

### Step 5: QA 검증

Merge Gate 통과 후 QA 에이전트에게 경계면 검증 요청:
- `.claude/agents/qa.md`의 체크리스트 실행
- 발견 사항을 다음 라운드에 반영

### Step 6: 루프 반복

목표 score 도달 또는 WBS 완료까지 Step 1로 돌아간다.

## 에러 핸들링

| 상황 | 대응 |
|------|------|
| 워커 빌드 실패 | 해당 워커 결과 버리고 다음 워커로 |
| score 하락 | 즉시 롤백, 원인 분석 후 재시도 |
| 테스트 실패 | QA에게 위임하여 원인 파악 |
| 머지 충돌 | 모듈 소유권 위반 — 충돌 워커 결과 버림 |

## 테스트 시나리오

**정상 흐름**: score 0.90 → architect가 perf-dev에 NEON 최적화 위임 → simd_speedup 4x → merge gate 통과 → score 0.95

**에러 흐름**: core-dev의 polar 변경이 turbo 테스트를 깨뜨림 → merge gate에서 score 하락 감지 → 롤백 → qa가 경계면 불일치 보고 → core-dev에 수정 지시
