---
name: Bug Report
about: Something isn't working as expected
title: ''
labels: bug
assignees: ''
---

**Model & Config**
- Model: (e.g., Llama-3.2-3B-Instruct-Q8_0.gguf)
- KV compression: (e.g., `-k uniform_4b -v q4`)
- Platform: (e.g., macOS M1 Pro 16GB)

**What happened?**
A clear description of the bug.

**Expected behavior**
What you expected to happen.

**Steps to reproduce**
```bash
./build/quant model.gguf -p "..." -n 50
```

**Output**
```
(paste output here)
```

**Build info**
```bash
git log --oneline -1
cmake --build build 2>&1 | grep -c "warning:"
```
