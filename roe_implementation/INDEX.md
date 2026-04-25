# RoE Implementation File Index

This folder contains a complete implementation of RoE (Repetition of Experts) for Open-RAG.

## 📁 File Structure

```
roe_implementation/
├── README.md                    ⭐ START HERE - Full documentation
├── SUMMARY.md                   ⭐ ANSWERS TO YOUR QUESTIONS
├── roe_openrag.py              🔧 Core RoE implementation
├── modeling_openrag_roe.py     🔧 Modified MoE components
├── example_integration.py      📖 Integration examples & tutorials
├── test_roe.py                 🧪 Unit tests
├── compare_results.py          📊 Results comparison tool
└── INDEX.md                    📋 This file
```

## 🚀 Quick Start Guide

### Step 1: Understand the Concepts
**Read first:** `SUMMARY.md`
- Answers all your questions about RoE
- Explains probability averaging
- Shows why clean cache is needed
- Compares RoE vs top-p sampling

### Step 2: Learn the API
**Read next:** `README.md`
- Full documentation
- API reference
- Parameter explanations
- Expected results

### Step 3: Test the Implementation
**Run:** `test_roe.py`
```bash
cd roe_implementation
python test_roe.py
```
This verifies:
- ✓ Context manager works
- ✓ Router modification works
- ✓ Probability averaging works
- ✓ KV cache broadcasting works

### Step 4: See Integration Examples
**Read:** `example_integration.py`
```bash
python example_integration.py
```
Shows:
- How to integrate into your pipeline
- Command-line argument setup
- Visual explanation of RoE process

### Step 5: Integrate into Your Code
**Modify:** `run_short_form_multihop.py`

See `example_integration.py` for exact code to add.

### Step 6: Compare Results
**Run:** `compare_results.py`
```bash
# After running both baseline and RoE
python compare_results.py baseline.jsonl roe.jsonl
```

## 📚 Detailed File Descriptions

### `README.md` ⭐
**Purpose:** Complete documentation of RoE implementation

**Contents:**
- What is RoE and why it works
- Quick start guide
- Integration options
- Parameter explanations
- Expected results
- Troubleshooting

**When to read:** Before integrating RoE into your code

---

### `SUMMARY.md` ⭐
**Purpose:** Comprehensive answers to your questions

**Contents:**
- Q1: Controlled variation explanation
- Q2: Probability averaging explained with examples
- Q3: RoE vs top-p sampling comparison
- Q4: Why tau=0 for clean cache
- Q5: Context manager pattern
- Q6: KV cache broadcasting
- Q7: Clean cache necessity

**When to read:** To understand the concepts deeply

---

### `roe_openrag.py` 🔧
**Purpose:** Core RoE implementation

**Key functions:**
- `roe_context()`: Context manager for RoE state
- `get_roe_state()`: Retrieve current RoE configuration
- `apply_roe_to_router()`: Apply temperature to router logits
- `generate_with_roe()`: Main generation function
- `create_default_tau_map()`: Helper to create tau map

**Usage:**
```python
from roe_openrag import generate_with_roe, create_default_tau_map

tau_map = create_default_tau_map(num_layers, tau_middle=1.0)
text = generate_with_roe(model, tokenizer, prompt, K=8, layer_tau_map=tau_map)
```

---

### `modeling_openrag_roe.py` 🔧
**Purpose:** Modified MoE components with RoE support

**Key classes:**
- `OpenRAGGateAdapterRoE`: Router with RoE temperature support
- `LlamaMLPRoE`: MLP with RoE-enabled adapter
- `ParallelAdapterMLPRoE`: Adapter with RoE awareness

**When to use:**
- If you want to integrate RoE directly into model architecture
- For extensive experimentation
- **Not needed** for wrapper approach (recommended)

**Documentation:**
- Includes detailed explanations as comments
- Shows how RoE differs from baseline
- Explains key concepts (Q&A format at bottom)

---

### `example_integration.py` 📖
**Purpose:** Practical integration examples

**Functions:**
- `integrate_roe_into_existing_pipeline()`: Full working example
- `modify_existing_generate_function()`: Step-by-step guide
- `visualize_roe_process()`: Visual explanation

**When to use:**
- When integrating RoE into your code
- To understand the generation process visually
- To see expected command-line arguments

**Run it:**
```bash
python example_integration.py
# Shows integration guide and visual explanation
```

---

### `test_roe.py` 🧪
**Purpose:** Unit tests to verify implementation

**Tests:**
1. RoE context manager (set/reset state)
2. Router modification (tau=0 vs tau>0)
3. Probability averaging (correct math)
4. KV cache broadcasting (shape compatibility)
5. Tau map creation (correct defaults)

**Run it:**
```bash
python test_roe.py
```

**Expected output:**
```
TEST 1: RoE Context Manager
✓ Default state correct
✓ Context manager sets state correctly
✓ Context manager cleans up correctly
✅ TEST 1 PASSED

... (more tests)

🎉 ALL TESTS PASSED!
```

---

### `compare_results.py` 📊
**Purpose:** Compare baseline vs RoE results

**Features:**
- Metric comparison (EM, F1, Precision, Recall)
- Prediction comparison (side-by-side)
- Disagreement analysis (where they differ)
- Retrieval decision analysis

**Usage:**
```bash
# Run baseline
python run_short_form_multihop.py --task hotpotqa --output_file baseline.jsonl

# Run RoE
python run_short_form_multihop.py --task hotpotqa --use_roe --roe_k 8 --output_file roe.jsonl

# Compare
python compare_results.py baseline.jsonl roe.jsonl --num_examples 10
```

**Output:**
- Overall score comparison
- Per-metric breakdown
- Examples where RoE differs
- Analysis of improvements/regressions

---

## 🎯 Reading Order by Goal

### Goal: Understand RoE Concepts
1. `SUMMARY.md` - Read all Q&As
2. `README.md` - Skim "What is RoE" section
3. `example_integration.py` - Run `visualize_roe_process()`

### Goal: Quick Integration
1. `README.md` - Read "Quick Start"
2. `example_integration.py` - Copy integration code
3. `test_roe.py` - Run to verify

### Goal: Deep Understanding
1. `SUMMARY.md` - Read thoroughly
2. `roe_openrag.py` - Read implementation
3. `modeling_openrag_roe.py` - Read Q&A at bottom
4. `test_roe.py` - Study each test

### Goal: Evaluate RoE
1. Integrate using `example_integration.py`
2. Run experiments
3. Use `compare_results.py` to analyze

## 🔍 Key Code Snippets

### Minimal Integration (Wrapper Approach)
```python
from roe_implementation.roe_openrag import generate_with_roe, create_default_tau_map

# After loading model
tau_map = create_default_tau_map(model.config.num_hidden_layers)

# Replace model.generate() with:
text = generate_with_roe(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=50,
    K=8,
    layer_tau_map=tau_map,
    device="cuda"
)
```

### Testing RoE Works
```python
# Run tests
python test_roe.py

# Should see:
# ✅ TEST 1 PASSED
# ✅ TEST 2 PASSED
# ...
# 🎉 ALL TESTS PASSED!
```

### Comparing Results
```bash
python compare_results.py baseline.jsonl roe.jsonl

# Shows:
# Baseline: 0.4523
# RoE:      0.4891
# Improvement: +8.14%
```

## 📞 Need Help?

### If tests fail:
- Check `test_roe.py` error messages
- Verify torch and transformers are installed
- Check Python version (3.8+)

### If integration fails:
- Review `example_integration.py`
- Check model path is correct
- Verify CUDA is available

### If results are worse:
- Try different K values (4, 8, 16)
- Adjust tau_middle (0.5, 1.0, 2.0)
- Check if model supports trust_remote_code

### If generation is too slow:
- Reduce K (e.g., K=4)
- Use fp16/bf16
- Set tau=0 for more layers

## 🎓 Learning Path

**Beginner:** Just want to use RoE
1. Read `README.md` Quick Start
2. Run `test_roe.py`
3. Copy code from `example_integration.py`

**Intermediate:** Want to understand RoE
1. Read `SUMMARY.md` thoroughly
2. Study `roe_openrag.py` implementation
3. Experiment with different tau values

**Advanced:** Want to extend RoE
1. Study `modeling_openrag_roe.py`
2. Modify router temperature strategy
3. Try adaptive tau selection

## 📝 Notes

- **Wrapper approach** (using `generate_with_roe()`) is **recommended**
  - Cleaner code
  - Easier to maintain
  - No need to modify model architecture
  
- **Direct integration** (using `modeling_openrag_roe.py`) only if:
  - Doing extensive research
  - Need fine-grained control
  - Want to modify RoE algorithm

- **Performance**: RoE is ~K times slower but potentially better quality
  - K=4: Good for experimentation
  - K=8: Recommended for evaluation
  - K=16: Best quality, very slow

## ✅ Checklist

Before running experiments:
- [ ] Read `SUMMARY.md` to understand concepts
- [ ] Run `test_roe.py` to verify implementation
- [ ] Review `example_integration.py` for integration
- [ ] Choose appropriate K and tau_map
- [ ] Run baseline for comparison

During experiments:
- [ ] Monitor GPU memory usage
- [ ] Save both baseline and RoE results
- [ ] Use same random seed for fair comparison
- [ ] Track generation time

After experiments:
- [ ] Use `compare_results.py` to analyze
- [ ] Check metric improvements
- [ ] Analyze disagreement cases
- [ ] Document findings

## 🎉 You're Ready!

Everything you need to implement and evaluate RoE is in this folder. Start with `SUMMARY.md` to understand the concepts, then follow the integration guide in `example_integration.py`.

Good luck with your experiments! 🚀
