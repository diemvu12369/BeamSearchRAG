# RoE (Repetition of Experts) for Open-RAG

Implementation of "MoEs Are Stronger Than You Think: Hyper-Parallel Inference Scaling with RoE" for the Open-RAG model.

## 📖 What is RoE?

**RoE (Repetition of Experts)** is a technique to improve the generation quality of Mixture-of-Experts (MoE) models by running inference **K times in parallel** with different expert selections, then aggregating the results.

### Key Idea

Instead of generating each token once:
```
Input -> MoE Layer (select experts) -> Output token
```

RoE generates each token **K times** with different expert combinations:
```
Input -> MoE Layer (experts set 1) -> Output 1
      -> MoE Layer (experts set 2) -> Output 2
      ...
      -> MoE Layer (experts set K) -> Output K

Then: Average probabilities from all K outputs -> Final token
```

## 🎯 Why RoE Works

### 1. **Diversity through Stochastic Expert Selection**
- Each of the K forward passes may route to different experts
- Controlled by temperature parameter τ (tau)
- τ = 0: Deterministic (same experts every time)
- τ > 0: Stochastic (different experts via Gumbel noise)

### 2. **Robustness through Probability Averaging**
- Each forward pass produces a probability distribution over vocab
- Average all K distributions: `avg_probs = mean(probs_1, probs_2, ..., probs_K)`
- Select token with highest averaged probability
- Outlier predictions get averaged out

### 3. **Consistency through Clean Cache**
- KV cache is updated using only the deterministic path (τ=0)
- Ensures generation stability across steps
- Exploration only affects current token prediction, not history

## 🚀 Quick Start

### Installation

```bash
cd openrag/roe_implementation
# All required dependencies are already in your environment
```

### Basic Usage

```python
from roe_openrag import generate_with_roe, create_default_tau_map
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "shayekh/openrag_llama2_7b_8x135m",
    device_map="cuda:0",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("shayekh/openrag_llama2_7b_8x135m")

# Create tau map (temperature for each layer)
num_layers = model.config.num_hidden_layers
tau_map = create_default_tau_map(num_layers, tau_middle=1.0)

# Generate with RoE
text = generate_with_roe(
    model=model,
    tokenizer=tokenizer,
    prompt="What is the capital of France?",
    max_new_tokens=50,
    K=8,  # Number of parallel samples
    layer_tau_map=tau_map,
    device="cuda:0"
)

print(text)
```

## 📊 How to Integrate into Your Pipeline

### Option 1: Wrapper Approach (Recommended)

Modify your `run_short_form_multihop.py`:

```python
# Add imports
from roe_implementation.roe_openrag import generate_with_roe, create_default_tau_map

# After loading model
tau_map = create_default_tau_map(model.config.num_hidden_layers, tau_middle=1.0)

# Replace model.generate() calls with:
text = generate_with_roe(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=max_new_tokens,
    K=8,
    layer_tau_map=tau_map,
    device="cuda"
)
```

### Option 2: Command-Line Flag

Add RoE as an optional feature:

```python
# In argparse
parser.add_argument("--use_roe", action="store_true")
parser.add_argument("--roe_k", type=int, default=8)

# In generation
if args.use_roe:
    text = generate_with_roe(model, tokenizer, prompt, K=args.roe_k, ...)
else:
    outputs = model.generate(...)
```

Run with:
```bash
python run_short_form_multihop.py \
    --model_name shayekh/openrag_llama2_7b_8x135m \
    --task hotpotqa \
    --use_roe \
    --roe_k 8
```

## 🔧 Key Parameters

### `K` (Number of Parallel Samples)
- Higher K = better quality but slower
- Paper uses K=4-16
- Trade-off: K=8 gives ~8x slower generation but better accuracy

### `tau_map` (Temperature Map)
- Dictionary: `{layer_idx: temperature}`
- τ=0: Deterministic expert selection
- τ>0: Stochastic expert selection
- Paper's heuristic: τ=0 for first/last layers, τ=1.0 for middle layers

```python
# Example tau_map for 32-layer model
tau_map = {
    0: 0.0,      # First layer: deterministic
    1: 1.0,      # Middle layers: explore
    2: 1.0,
    ...
    30: 1.0,
    31: 0.0,     # Last layer: deterministic
}
```

## 📈 Expected Results

### Benefits
1. **Better Answer Quality**: Ensemble effect from multiple expert paths
2. **More Robust Predictions**: Outliers averaged out
3. **Improved Retrieval Decisions**: Better [Retrieval] vs [No Retrieval] accuracy

### Trade-offs
- **Speed**: ~K times slower per token
- **Memory**: Same (no extra parameters)
- **Quality**: Higher (ensemble of experts)

## 🧠 Understanding the Concepts

### Q1: How does RoE differ from top-p sampling?

| Aspect | Top-p Sampling | RoE |
|--------|---------------|-----|
| Forward passes | 1 | K |
| Diversity source | Probabilistic token selection | Multiple expert combinations |
| Aggregation | None | Probability averaging |
| Determinism | Stochastic | Can be deterministic (greedy on averaged probs) |

### Q2: Why do we need a "clean cache"?

The KV cache stores the history of generated tokens. If we updated it with stochastic expert selections:
- Cache would depend on random choices
- Generation would be unstable
- Different runs would diverge significantly

Solution: Always update cache with deterministic path (τ=0)

### Q3: How does shared KV cache work with K batched inputs?

HuggingFace transformers broadcast the KV cache automatically:
- `past_key_values`: [batch=1, num_heads, seq_len, head_dim]
- `step_input`: [batch=K, 1]
- PyTorch broadcasting handles shape alignment
- All K samples share the same history, only current token differs

## 📝 Code Structure

```
roe_implementation/
├── roe_openrag.py              # Core RoE implementation
│   ├── roe_context()           # Context manager for RoE state
│   ├── apply_roe_to_router()   # Apply temperature to router logits
│   └── generate_with_roe()     # Main generation function
│
├── modeling_openrag_roe.py     # Modified MoE components
│   ├── OpenRAGGateAdapterRoE   # Router with RoE support
│   └── LlamaMLPRoE             # MLP with RoE-enabled adapter
│
├── example_integration.py      # Integration examples
│   ├── integrate_roe_into_existing_pipeline()
│   ├── modify_existing_generate_function()
│   └── visualize_roe_process()
│
└── README.md                   # This file
```

## 🎓 Paper References

The concepts mentioned in the paper excerpt:

### "Controlled variation within each transformer block"
- **What**: Run the same block multiple times with different configurations
- **How**: RoE varies expert selection via temperature τ
- **Why**: Explores different computational paths without changing architecture

### "Reuse each layer repeatedly in a recurrent manner"
- **What**: Pass hidden states through layers multiple times
- **How**: Not the focus of RoE; RoE focuses on parallel exploration
- **Why**: Increases computation without adding parameters

### "Hyper-parallel scaling"
- **What**: Increase computation per token at inference time
- **How**: K parallel forward passes with different expert selections
- **Why**: Unlock model's full potential through ensemble

## 🔬 Experimental Tips

### Recommended Settings for Open-RAG

**Important:** For Open-RAG, use small tau values (0.01-0.1). Large values degrade quality!

```python
# Recommended: Random tau for each layer (more diversity)
K = 8
tau_map = create_default_tau_map(num_layers, tau_range=(0.0, 0.01))

# Alternative: Fixed small tau
K = 8
tau_map = create_default_tau_map(num_layers, tau_middle=0.05)

# For shorter answers (arc_c, fever)
K = 4
tau_map = create_default_tau_map(num_layers, tau_range=(0.0, 0.01))

# For faster experimentation
K = 2
tau_map = create_default_tau_map(num_layers, tau_middle=0.01)
```

#### Tau Sampling Strategies

1. **Random tau per layer** (recommended for diversity):
   ```python
   # Each middle layer gets random tau in [0, 0.01]
   tau_map = create_default_tau_map(num_layers, tau_range=(0.0, 0.01))
   ```

2. **Fixed tau** (consistent across layers):
   ```python
   # All middle layers use tau=0.05
   tau_map = create_default_tau_map(num_layers, tau_middle=0.05)
   ```

3. **Custom per-layer control**:
   ```python
   # Manually set tau for each layer
   tau_map = {
       0: 0.0,      # First layer always 0
       1: 0.02,
       2: 0.01,
       ...
       31: 0.0      # Last layer always 0
   }
   ```

### Measuring Impact

Compare metrics with/without RoE:

```bash
# Baseline
python run_short_form_multihop.py --task hotpotqa --output_file baseline.jsonl

# With RoE
python run_short_form_multihop.py --task hotpotqa --use_roe --roe_k 8 --output_file roe.jsonl

# Compare
python compare_results.py baseline.jsonl roe.jsonl
```

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce K or use gradient checkpointing

```python
# Reduce K
K = 4  # instead of 8

# Or process in smaller batches
# (RoE already processes K samples, but you can reduce prompt batch size)
```

### Issue: Slower than expected

**Expected**: K=8 means ~8x slower per token

**Tips to speed up**:
1. Reduce K to 4
2. Lower tau for some layers (less exploration)
3. Use mixed precision (fp16)

### Issue: No quality improvement

**Check**:
1. Is tau > 0 for middle layers?
2. Is K large enough (try K=8)?
3. Are you using greedy decoding on averaged probs?

## 📚 Additional Resources

- Original Paper: "MoEs Are Stronger Than You Think: Hyper-Parallel Inference Scaling with RoE"
- Open-RAG Paper: [Link to Open-RAG paper]
- HuggingFace Transformers Docs: https://huggingface.co/docs/transformers/

## 🤝 Contributing

If you improve the implementation or find bugs, feel free to:
1. Document your changes
2. Test on multiple tasks
3. Share results!

## 📄 License

Same as Open-RAG repository.

---

**Questions?** Check `example_integration.py` for detailed explanations and visual diagrams!
