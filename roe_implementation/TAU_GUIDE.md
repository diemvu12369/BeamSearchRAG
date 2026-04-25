# Tau Configuration Guide for Open-RAG RoE

## TL;DR

**Use small tau values for Open-RAG!**

✅ **Recommended:** `tau_range=(0.0, 0.01)` - Each layer gets random tau between 0 and 0.01
✅ **Alternative:** `tau_middle=0.05` - All layers use fixed tau of 0.05
❌ **Avoid:** `tau > 0.1` - Too much noise, degrades quality

## Why Small Tau?

Unlike the original RoE paper (which uses tau=1.0), Open-RAG's router is more sensitive to noise. Large tau values cause the router to select suboptimal experts, producing gibberish.

## Usage Examples

### 1. Random Tau Per Layer (Most Diverse)

```python
from roe_openrag import create_default_tau_map

# Each middle layer gets a different random tau in [0, 0.01]
tau_map = create_default_tau_map(num_layers=32, tau_range=(0.0, 0.01))

# Example output:
# {
#   0: 0.0,      # First layer: deterministic
#   1: 0.0043,   # Random
#   2: 0.0089,   # Random
#   3: 0.0012,   # Random
#   ...
#   30: 0.0067,  # Random
#   31: 0.0      # Last layer: deterministic
# }
```

**When to use:** Maximum diversity, best for multi-hop reasoning tasks like HotpotQA.

### 2. Fixed Small Tau (Consistent)

```python
# All middle layers use tau=0.05
tau_map = create_default_tau_map(num_layers=32, tau_middle=0.05)

# Output:
# {
#   0: 0.0,    # First layer: deterministic
#   1: 0.05,   # All middle layers
#   2: 0.05,
#   ...
#   30: 0.05,
#   31: 0.0    # Last layer: deterministic
# }
```

**When to use:** Consistent behavior across layers, good for ablation studies.

### 3. Custom Per-Layer Control

```python
# Manually specify tau for each layer
tau_map = {
    0: 0.0,      # First layer always deterministic
    1: 0.01,     # Early layers: small tau
    2: 0.01,
    ...
    15: 0.05,    # Middle layers: slightly higher
    16: 0.05,
    ...
    30: 0.01,    # Late layers: small tau
    31: 0.0      # Last layer always deterministic
}
```

**When to use:** Fine-grained control, layer-specific tuning.

## Tau Value Guidelines

| Tau Range | Behavior | Use Case |
|-----------|----------|----------|
| 0.0 | Deterministic (no exploration) | Baseline comparison |
| 0.001 - 0.01 | Very light exploration | Conservative, stable |
| 0.01 - 0.05 | Light exploration | **Recommended default** |
| 0.05 - 0.1 | Moderate exploration | Careful, may degrade quality |
| > 0.1 | Heavy exploration | ❌ Too noisy for Open-RAG |

## Integration with run_short_form_multihop_roe.py

The script supports both approaches:

```bash
# Option 1: Use default (random tau in [0, 0.01])
python run_short_form_multihop_roe.py \
    --use_roe \
    --roe_k 8 \
    --roe_tau 0.05

# Option 2: Can be modified in code to use tau_range
# Edit the script to use:
#   tau_map = create_default_tau_map(num_layers, tau_range=(0.0, 0.01))
```

## Testing Different Tau Strategies

Use the provided test script:

```bash
cd roe_implementation
python test_tau_values.py
```

This will test:
- Fixed tau=0.0 (deterministic baseline)
- Fixed tau=0.01
- Fixed tau=0.05
- Random tau in [0, 0.01]
- Random tau in [0, 0.05]
- Fixed tau=0.1

Compare the outputs to find what works best for your use case!

## Common Issues

### Issue: Generated text is gibberish
**Cause:** Tau is too large
**Solution:** Reduce tau to 0.01-0.05 or use tau_range=(0.0, 0.01)

### Issue: No improvement over baseline
**Cause:** Tau is 0 (no exploration) or K is too small
**Solution:** Use tau_range=(0.0, 0.01) and K=8

### Issue: Too slow
**Cause:** K is large
**Solution:** Reduce K to 4 or 2, but expect less improvement

## Summary

1. **Always use small tau for Open-RAG** (0.01-0.05)
2. **Random tau per layer** adds more diversity than fixed tau
3. **Start with:** `tau_range=(0.0, 0.01)` and `K=8`
4. **If too slow:** Reduce K to 4
5. **If quality issues:** Reduce tau to (0.0, 0.01)
