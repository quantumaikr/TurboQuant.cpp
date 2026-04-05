## What does this PR do?

(Brief description)

## Checklist

- [ ] `cmake --build build` — zero warnings
- [ ] `ctest --test-dir build` — 34/34 pass
- [ ] No files modified in `refs/`
- [ ] README updated (if user-facing change)

## Test plan

How did you verify this works?

```bash
./build/quant model.gguf -p "test" -n 10
```
