---
name: develop
description: "TurboQuant.cpp의 다음 WBS 항목을 자율적으로 구현한다. Karpathy 루프 패턴: score → implement → score → commit/revert. '개발해줘', '다음 항목 구현', '다음 WBS', 'develop', 특정 모듈명(polar, qjl 등) 언급 시 사용."
argument-hint: "모듈명 (예: polar, qjl, cache) 또는 빈칸 (자동 선택)"
---

# Develop — Autonomous Single-Item Implementation

Karpathy AutoResearch 패턴으로 WBS 항목을 하나씩 구현한다.

## 프로토콜

### 1. 상태 평가
```bash
bash score.sh --quick
```
WBS 파일에서 다음 미완료 항목 `- [ ]`을 찾는다. 인자가 있으면 해당 모듈만.

### 2. 참조 코드 읽기

구현 전 반드시 `.claude/agents/core-dev.md`의 "참조 코드 매핑" 테이블에서 해당 알고리즘의 원본을 읽는다. 이유: refs/ 코드가 정답이며, 우리는 이를 C로 포팅하는 것이다.

### 3. 구현

- CLAUDE.md의 모듈 소유권 테이블을 준수한다
- 하나의 WBS 항목만 구현한��� (작고 정확하게)
- edge case 방어: NULL 체크, 범위 검증, 오버플로 방지

### 4. 검증
```bash
bash score.sh --quick
```
- score 상승/유지 → 진행
- score 하락 → `git checkout -- .` 후 다른 접근법 시도

### 5. 커밋

WBS 항목을 `[x]`로 마크하고, 변경된 파일만 stage하여 커밋.

## 규칙
- `refs/`, `program.md`, `score.sh`는 수정 금지
- 한 번에 하나의 WBS 항목만
- 모든 테스트 통과 확인 후 커밋
