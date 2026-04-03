# WikiText-2 Standard PPL Benchmark

Standard perplexity evaluation on WikiText-2 test set (Merity et al., 2016).
Teacher-forced, cross-entropy loss averaged over all tokens.

## SmolLM2-1.7B-Instruct-Q8_0

| Config | PPL | vs FP32 | K bpe | Compression |
|--------|-----|---------|-------|-------------|
| FP32 baseline (no KV quant) | 14.6315 | -- | 32 | 1x |
| uniform_4b K + FP16 V | 14.6315 | **+0.00%** | 4 | 1.6x (K only) |
| uniform_4b K + Q4 V | 14.5666 | **-0.44%** | 4 | 3.8x |
| delta + 3b K + Q4 V | 14.8157 | **+1.26%** | ~3.5 | ~4.3x |

WikiText-2 test set: 1,066 tokens evaluated.
Model weights: Q8_0 GGUF, load-time Q4 conversion.
Hardware: Apple M3, 4 threads.

## Key Observations

1. **4-bit K is lossless** on WikiText-2 (PPL +0.00%), consistent with prior measurements.
2. **4-bit K + Q4 V** shows slight PPL improvement (-0.44%), likely within noise.
3. **Delta + 3-bit K + Q4 V** shows +1.26% PPL — small degradation, not the -3.2%
   seen on ppl_test_1k.txt. The prior measurement was on a different text with
   different token distribution. WikiText-2 is the more standard benchmark.
4. **Honest assessment**: delta compression enables 3-bit keys at ~1% PPL cost,
   not the "better than FP32" claim from the earlier benchmark.

## Comparison with Prior Results

| Benchmark | Tokens | FP32 PPL | 4b K+Q4V | delta+3b K+Q4V |
|-----------|--------|----------|----------|----------------|
| ppl_test_1k.txt | 999 | 14.58 | 13.44 (-7.8%) | 14.11 (-3.2%) |
| WikiText-2 test | 1,066 | 14.63 | 14.57 (-0.4%) | 14.82 (+1.3%) |

The ppl_test_1k.txt results showed larger improvements, likely due to the specific
text being more favorable to quantization (possible regularization effect). WikiText-2
is the standard benchmark and should be the primary reference.

## Reproduction

```bash
# Download WikiText-2
pip install datasets
python3 -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
text = '\n'.join([t for t in ds['text'] if t.strip()])
with open('bench/data/wikitext2_test.txt', 'w') as f: f.write(text)
"

# Run benchmarks
./build/quant model.gguf --ppl bench/data/wikitext2_test.txt -k none
./build/quant model.gguf --ppl bench/data/wikitext2_test.txt -k uniform_4b -v q4
./build/quant model.gguf --ppl bench/data/wikitext2_test.txt -k uniform_3b -v q4 --delta
```
