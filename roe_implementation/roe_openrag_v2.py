"""
RoE (Repetition of Experts) Implementation for Open-RAG - Version 2
Uses model.generate() properly instead of manual forward passes
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from contextlib import contextmanager
from threading import local

# Global thread-local storage for RoE state
_roe_state = local()


@contextmanager
def roe_context(layer_tau_map: Dict[int, float], K: int = 1):
    """
    Context manager to control RoE behavior during generation.
    """
    old_map = getattr(_roe_state, 'layer_tau_map', None)
    old_K = getattr(_roe_state, 'K', 1)
    
    _roe_state.layer_tau_map = layer_tau_map
    _roe_state.K = K
    
    try:
        yield
    finally:
        _roe_state.layer_tau_map = old_map
        _roe_state.K = old_K


def get_roe_state():
    """Get current RoE state"""
    layer_tau_map = getattr(_roe_state, 'layer_tau_map', None)
    K = getattr(_roe_state, 'K', 1)
    return layer_tau_map, K


def apply_roe_to_router(router_logits: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Apply Gumbel noise to router logits based on tau"""
    layer_tau_map, K = get_roe_state()
    
    if layer_tau_map is None:
        return router_logits
    
    tau = layer_tau_map.get(layer_idx, 0.0)
    
    if tau > 0:
        gumbel_noise = torch.nn.functional.gumbel_softmax(
            torch.zeros_like(router_logits),
            tau=1.0,
            hard=False
        ).log()
        router_logits = router_logits + tau * gumbel_noise
    
    return router_logits


def patch_model_for_roe(model):
    """Monkey-patch the model's router to use RoE"""
    num_patched = 0
    
    for name, module in model.named_modules():
        if 'gate' in name.lower() and hasattr(module, 'forward'):
            original_forward = module.forward
            layer_idx = num_patched
            
            def make_patched_forward(orig_forward, idx):
                def patched_forward(hidden_states):
                    logits = orig_forward(hidden_states)
                    return apply_roe_to_router(logits, idx)
                return patched_forward
            
            module.forward = make_patched_forward(original_forward, layer_idx)
            num_patched += 1
    
    print(f"[RoE] Patched {num_patched} router layers")
    return model


@torch.no_grad()
def generate_with_roe(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    K: int = 8,
    layer_tau_map: Optional[Dict[int, float]] = None,
    device: str = "cuda",
    **generate_kwargs
):
    """
    Generate with RoE by running model.generate() K times and averaging probabilities.
    
    This approach:
    1. Runs model.generate() K times with different tau values (stochastic routing)
    2. Collects logits/scores at each position for each run
    3. Averages probabilities across K runs
    4. Decodes the most likely sequence from averaged probabilities
    
    Args:
        model: Patched OpenRAG model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Max tokens to generate
        K: Number of parallel samples
        layer_tau_map: tau values for each layer
        device: Device
        **generate_kwargs: Additional arguments for model.generate()
    
    Returns:
        Generated text
    """
    if layer_tau_map is None:
        num_layers = model.config.num_hidden_layers
        layer_tau_map = create_default_tau_map(num_layers, tau_range=(0.0, 0.01))
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    
    print(f"[RoE] Generating with K={K} samples")
    print(f"[RoE] Input length: {input_len}, max_new_tokens: {max_new_tokens}")
    
    # Collect logits from K runs
    all_logits = []  # List of [seq_len, vocab_size] tensors
    
    for k in range(K):
        print(f"[RoE] Sample {k+1}/{K}...")
        
        with roe_context(layer_tau_map, K=1):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                **generate_kwargs
            )
        
        # outputs.scores is a tuple of tensors, each [batch_size, vocab_size]
        # Stack them: [seq_len, batch_size, vocab_size]
        logits = torch.stack(outputs.scores, dim=0)[:, 0, :]  # [seq_len, vocab_size]
        all_logits.append(logits)
    
    # Stack: [K, seq_len, vocab_size]
    all_logits = torch.stack(all_logits, dim=0)
    
    print(f"[RoE] Averaging probabilities across {K} samples...")
    
    # Convert to probabilities and average
    probs = F.softmax(all_logits, dim=-1)  # [K, seq_len, vocab_size]
    avg_probs = probs.mean(dim=0)  # [seq_len, vocab_size]
    
    # Select tokens greedily from averaged probabilities
    generated_ids = avg_probs.argmax(dim=-1)  # [seq_len]
    
    # Decode
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"[RoE] Generation complete!")
    return generated_text


def create_default_tau_map(num_layers: int, tau_middle: float = 0.05, tau_range=None):
    """Create tau map for layers"""
    tau_map = {}
    
    if tau_range is not None:
        tau_min, tau_max = tau_range
        for i in range(num_layers):
            if i == 0 or i == num_layers - 1:
                tau_map[i] = 0.0
            else:
                tau_map[i] = torch.rand(1).item() * (tau_max - tau_min) + tau_min
    else:
        for i in range(num_layers):
            if i == 0 or i == num_layers - 1:
                tau_map[i] = 0.0
            else:
                tau_map[i] = tau_middle
    
    return tau_map


# Test function
def test_roe():
    """Test RoE with Open-RAG"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "shayekh/openrag_llama2_7b_8x135m"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).eval()
    
    # Patch model
    model = patch_model_for_roe(model)
    
    # Create tau map
    num_layers = model.config.num_hidden_layers
    tau_map = create_default_tau_map(num_layers, tau_range=(0.0, 0.0))
    
    prompt = "[INST]What is the capital of France?[/INST]"
    
    print("\n" + "="*60)
    print("Generating with RoE...")
    print("="*60)
    
    generated_text = generate_with_roe(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        K=1,
        layer_tau_map=tau_map,
        device="cuda:0"
    )
    
    print("\n" + "="*60)
    print("Generated text:")
    print(generated_text)
    print("="*60)


if __name__ == "__main__":
    test_roe()
