"""
RoE (Repetition of Experts) Implementation for Open-RAG
Based on "MoEs Are Stronger Than You Think: Hyper-Parallel Inference Scaling with RoE"

Key Concepts:
1. Hyper-parallel scaling: Run inference K times with different expert selections
2. Probability averaging: Aggregate predictions across K runs
3. Clean cache: Maintain KV cache with deterministic (tau=0) expert selection
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from contextlib import contextmanager
from threading import local

# Global thread-local storage for RoE state
_roe_state = local()


@contextmanager
def roe_context(layer_tau_map: Dict[int, float], K: int = 1):
    """
    Context manager to control RoE behavior during forward passes.
    
    Args:
        layer_tau_map: Mapping of layer_idx -> temperature tau
                       tau=0 means deterministic (clean path)
                       tau>0 means stochastic exploration
        K: Number of parallel samples (batch size multiplier)
    
    How it works:
    - Sets global state that routers can access
    - Router uses tau to add noise: logits = logits + tau * gumbel_noise
    - K controls how many times we duplicate the input
    """
    old_map = getattr(_roe_state, 'layer_tau_map', None)
    old_K = getattr(_roe_state, 'K', 1)
    
    # Set new state
    _roe_state.layer_tau_map = layer_tau_map
    _roe_state.K = K
    
    try:
        yield
    finally:
        # Restore old state
        _roe_state.layer_tau_map = old_map
        _roe_state.K = old_K


def get_roe_state() -> Tuple[Optional[Dict[int, float]], int]:
    """Get current RoE state (tau_map, K)"""
    layer_tau_map = getattr(_roe_state, 'layer_tau_map', None)
    K = getattr(_roe_state, 'K', 1)
    return layer_tau_map, K


def apply_roe_to_router(router_logits: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Apply RoE temperature to router logits.
    
    This function should be called in the OpenRAGGateAdapter.forward() method.
    
    Args:
        router_logits: [batch_size, num_experts] logits from router
        layer_idx: Which layer this router belongs to
    
    Returns:
        Modified router logits with Gumbel noise if tau > 0
    """
    layer_tau_map, K = get_roe_state()
    
    if layer_tau_map is None:
        return router_logits
    
    # Get temperature for this layer (default to 0 if not specified)
    tau = layer_tau_map.get(layer_idx, 0.0)
    
    if tau > 0:
        # Add Gumbel noise for stochastic expert selection
        # Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(router_logits) + 1e-10
        ))
        router_logits = router_logits + tau * gumbel_noise
    
    return router_logits


def patch_model_for_roe(model):
    """
    Monkey-patch the Open-RAG model to enable RoE in the router.
    
    This modifies the forward method of OpenRAGGateAdapter to apply
    RoE temperature to router logits.
    
    Args:
        model: The Open-RAG model to patch
    """
    print("[RoE] Patching model to enable RoE...")
    
    # Find all OpenRAGGateAdapter modules
    layer_idx = 0
    for name, module in model.named_modules():
        # Check if this is a gate adapter (router)
        if 'gate' in name.lower() and hasattr(module, 'weight'):
            original_forward = module.forward
            
            # Create a closure to capture layer_idx
            def create_patched_forward(layer_idx, original_forward):
                def patched_forward(hidden_states):
                    # Call original forward
                    router_logits = original_forward(hidden_states)
                    # Apply RoE
                    router_logits = apply_roe_to_router(router_logits, layer_idx)
                    return router_logits
                return patched_forward
            
            module.forward = create_patched_forward(layer_idx, original_forward)
            layer_idx += 1
    
    print(f"[RoE] Patched {layer_idx} router layers")
    return model


@torch.no_grad()
def generate_with_roe(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    K: int = 8,
    layer_tau_map: Optional[Dict[int, float]] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
    temperature: float = 1.0,
    use_greedy: bool = True
):
    """
    Generate text using RoE (Repetition of Experts).
    
    IMPORTANT: You must call patch_model_for_roe(model) before using this function!
    
    Algorithm:
    1. Encode prompt with clean path (tau=0) to build KV cache
    2. For each new token:
       a. Replicate input K times
       b. Forward with RoE (tau>0) to get K diverse predictions
       c. Average probabilities across K predictions
       d. Select next token from averaged distribution
       e. Update KV cache with clean path (tau=0)
    
    Args:
        model: OpenRAG model (MUST be patched with patch_model_for_roe first!)
        tokenizer: Tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        K: Number of parallel RoE samples
        layer_tau_map: {layer_idx: tau} for each MoE layer
                       Paper suggests tau=0 for first/last layers
        eos_token_id: End-of-sequence token
        device: Device to run on
        temperature: Sampling temperature for final token selection
        use_greedy: If True, use greedy decoding on averaged probs
    
    Returns:
        Generated text string
    """
    if layer_tau_map is None:
        # Default: Use tau=1.0 for all layers
        # You may want to set tau=0 for first/last MoE layers
        num_layers = model.config.num_hidden_layers
        layer_tau_map = {i: 1.0 for i in range(num_layers)}
    
    eos = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    
    # Encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    B0 = input_ids.size(0)  # Usually 1
    
    print(f"[RoE] Generating with K={K}, max_new_tokens={max_new_tokens}")
    print(f"[RoE] Layer tau map: {layer_tau_map}")
    
    # Step 1: Prime the clean cache with full prompt (tau=0, no exploration)
    print(f"[RoE] Step 1: Encoding prompt with clean path (tau=0)...")
    with roe_context(layer_tau_map={i: 0.0 for i in layer_tau_map.keys()}, K=1):
        out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
    
    generated = []
    
    # Step 2: Generate tokens one by one  
    for step in range(max_new_tokens):
        print(f"[RoE] Step {step}: Generating with K={K} parallel paths...")
        
        # CRITICAL: We DO NOT feed in a new token here!
        # When using past_key_values, the model will automatically predict the NEXT token
        # We just need to provide None or empty input_ids and let past_key_values do the work
        # But HuggingFace requires at least one token, so we use a dummy approach:
        # We'll do the full forward pass WITHOUT past_key_values
        # 
        # Actually, the correct approach for RoE is:
        # - We already have the KV cache from the prompt
        # - We run K forward passes, each with tau applied to get different expert selections
        # - Each forward pass uses the SAME KV cache but different expert routing
        # - We DO need to provide the full sequence so far (prompt + generated tokens)
        
        # Build the full sequence so far
        if generated:
            current_sequence = torch.cat([input_ids] + generated, dim=-1)  # [B0, prompt_len + num_generated]
        else:
            current_sequence = input_ids  # [B0, prompt_len]
        
        # Replicate for K samples
        step_input = current_sequence.expand(B0 * K, -1).contiguous()  # [K*B0, seq_len]
        
        # Forward with RoE exploration
        # DO NOT use past_key_values here! We need to recompute everything with different expert selections
        with roe_context(layer_tau_map, K=K):
            out = model(
                input_ids=step_input,
                use_cache=False
            )
        
        # Get logits: [K*B0, 1, vocab_size]
        logits = out.logits[:, -1, :]  # [K*B0, vocab_size]
        
        # Reshape to [B0, K, vocab_size]
        logits = logits.view(B0, K, -1)
        
        # Convert to probabilities
        probs = F.softmax(logits / temperature, dim=-1)  # [B0, K, vocab_size]
        
        # RoE aggregation: Average probabilities across K samples
        avg_probs = probs.mean(dim=1)  # [B0, vocab_size]
        
        # Select next token
        if use_greedy:
            next_token = avg_probs.argmax(dim=-1)  # [B0]
        else:
            # Sample from averaged distribution
            next_token = torch.multinomial(avg_probs, num_samples=1).squeeze(-1)
        
        # Append to output (need to unsqueeze to add batch dim)
        generated.append(next_token.unsqueeze(-1))  # [B0, 1]
        
        print(f"[RoE]   Selected token: {tokenizer.decode(next_token[0])}")
        
        # Check for EOS
        if (next_token == eos).all():
            print(f"[RoE] EOS reached at step {step}")
            break
    
    # Decode generated tokens
    if generated:
        gen_ids = torch.cat(generated, dim=-1)  # [B0, seq_len]
    else:
        gen_ids = torch.empty((B0, 0), dtype=torch.long, device=device)
    
    generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    
    print(f"[RoE] Generation complete!")
    return generated_text


def create_default_tau_map(num_layers: int, tau_middle: float = 0.05, tau_range: Optional[Tuple[float, float]] = None) -> Dict[int, float]:
    """
    Create default tau map following paper's heuristic:
    - First and last MoE layers: tau = 0 (deterministic)
    - Middle layers: tau sampled from range or set to tau_middle
    
    Note: For Open-RAG, tau should be small (0.01-0.05) to avoid too much noise.
    Large tau values (>0.1) cause the router to select poor experts.
    
    Args:
        num_layers: Total number of layers
        tau_middle: Temperature for middle layers (used if tau_range is None)
        tau_range: If provided, sample tau for each middle layer from (min, max) range
                   Example: (0.0, 0.01) samples tau between 0 and 0.01
    
    Returns:
        Dictionary mapping layer_idx -> tau
    """
    tau_map = {}
    
    # Use random sampling if tau_range is provided
    if tau_range is not None:
        tau_min, tau_max = tau_range
        for i in range(num_layers):
            if i == 0 or i == num_layers - 1:
                tau_map[i] = 0.0  # Deterministic for first/last
            else:
                # Sample tau uniformly from [tau_min, tau_max]
                tau_map[i] = torch.rand(1).item() * (tau_max - tau_min) + tau_min
    else:
        # Use fixed tau_middle for all middle layers
        for i in range(num_layers):
            if i == 0 or i == num_layers - 1:
                tau_map[i] = 0.0  # Deterministic for first/last
            else:
                tau_map[i] = tau_middle  # Stochastic for middle
    
    return tau_map


# Example usage function
def example_roe_generation():
    """
    Example of how to use RoE with Open-RAG model.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "shayekh/openrag_llama2_7b_8x135m"
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).eval()
    
    # IMPORTANT: Patch the model to enable RoE
    model = patch_model_for_roe(model)
    
    # Create tau map with random sampling between 0 and 0.01
    num_layers = model.config.num_hidden_layers
    tau_map = create_default_tau_map(num_layers, tau_range=(0.0, 0.0))
    
    # Use the correct prompt format for Open-RAG (Llama chat format)
    prompt = "[INST]What is the capital of France?[/INST]"
    
    print("\n" + "="*50)
    print("Generating with RoE...")
    print("="*50)
    
    generated_text = generate_with_roe(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        K=8,  # Use 8 parallel samples
        layer_tau_map=tau_map,
        device="cuda:0"
    )
    
    print("\n" + "="*50)
    print("Generated text:")
    print(generated_text)
    print("="*50)


if __name__ == "__main__":
    example_roe_generation()
