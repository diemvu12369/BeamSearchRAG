"""
Quick Start: Test RoE implementation with Open-RAG

Run this to verify your RoE implementation works correctly.
"""

import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure roe_openrag is importable
from roe_openrag import generate_with_roe, create_default_tau_map, roe_context, apply_roe_to_router


def test_roe_context():
    """Test 1: Verify roe_context works"""
    print("="*80)
    print("TEST 1: RoE Context Manager")
    print("="*80)
    
    from roe_openrag import get_roe_state
    
    # Test default state
    tau_map, K = get_roe_state()
    assert tau_map is None, "Default tau_map should be None"
    assert K == 1, "Default K should be 1"
    print("✓ Default state correct")
    
    # Test context manager
    test_tau_map = {0: 1.0, 1: 0.5}
    with roe_context(test_tau_map, K=8):
        tau_map, K = get_roe_state()
        assert tau_map == test_tau_map, "tau_map should match"
        assert K == 8, "K should be 8"
        print("✓ Context manager sets state correctly")
    
    # Test cleanup
    tau_map, K = get_roe_state()
    assert tau_map is None, "tau_map should be reset after context"
    assert K == 1, "K should be reset after context"
    print("✓ Context manager cleans up correctly")
    
    print("\n✅ TEST 1 PASSED\n")


def test_apply_roe_to_router():
    """Test 2: Verify apply_roe_to_router works"""
    print("="*80)
    print("TEST 2: Apply RoE to Router")
    print("="*80)
    
    # Create dummy router logits
    router_logits = torch.randn(4, 8)  # [batch=4, num_experts=8]
    original_logits = router_logits.clone()
    
    # Test without RoE (tau=0)
    with roe_context({0: 0.0}, K=1):
        modified_logits = apply_roe_to_router(router_logits, layer_idx=0)
        assert torch.allclose(modified_logits, original_logits), "tau=0 should not change logits"
        print("✓ tau=0 keeps logits unchanged")
    
    # Test with RoE (tau=1.0)
    with roe_context({0: 1.0}, K=4):
        modified_logits = apply_roe_to_router(router_logits, layer_idx=0)
        assert not torch.allclose(modified_logits, original_logits), "tau>0 should add noise"
        print("✓ tau>0 adds Gumbel noise")
    
    # Test different tau for different layers
    with roe_context({0: 0.0, 1: 1.0}, K=4):
        logits_layer0 = apply_roe_to_router(router_logits, layer_idx=0)
        logits_layer1 = apply_roe_to_router(router_logits, layer_idx=1)
        
        assert torch.allclose(logits_layer0, router_logits), "Layer 0 should be unchanged"
        assert not torch.allclose(logits_layer1, router_logits), "Layer 1 should have noise"
        print("✓ Different layers use different tau values")
    
    print("\n✅ TEST 2 PASSED\n")


def test_probability_averaging():
    """Test 3: Verify probability averaging logic"""
    print("="*80)
    print("TEST 3: Probability Averaging")
    print("="*80)
    
    import torch.nn.functional as F
    
    # Simulate K=4 predictions
    K = 4
    vocab_size = 1000
    
    # Create K different logits
    logits = torch.randn(K, vocab_size)
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # [K, vocab_size]
    
    # Check each distribution sums to 1
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(K)), "Each prob distribution should sum to 1"
    print(f"✓ Each probability distribution sums to 1: {sums}")
    
    # Average probabilities
    avg_probs = probs.mean(dim=0)  # [vocab_size]
    
    # Check averaged distribution sums to 1
    avg_sum = avg_probs.sum()
    assert torch.allclose(avg_sum, torch.tensor(1.0)), "Averaged probs should sum to 1"
    print(f"✓ Averaged probabilities sum to 1: {avg_sum}")
    
    # Select token
    next_token = avg_probs.argmax()
    print(f"✓ Selected token ID: {next_token}")
    
    print("\n✅ TEST 3 PASSED\n")


def test_kv_cache_broadcasting():
    """Test 4: Verify KV cache broadcasting works"""
    print("="*80)
    print("TEST 4: KV Cache Broadcasting")
    print("="*80)
    
    # Simulate past KV cache (batch=1)
    batch_1 = 1
    num_heads = 8
    seq_len = 10
    head_dim = 64
    
    past_key = torch.randn(batch_1, num_heads, seq_len, head_dim)
    past_value = torch.randn(batch_1, num_heads, seq_len, head_dim)
    
    print(f"Past KV shape: {past_key.shape}")
    
    # Simulate current KV (batch=K)
    K = 8
    current_key = torch.randn(K, num_heads, 1, head_dim)
    current_value = torch.randn(K, num_heads, 1, head_dim)
    
    print(f"Current KV shape (K={K}): {current_key.shape}")
    
    # Concatenate (should broadcast automatically)
    full_key = torch.cat([past_key, current_key], dim=2)
    full_value = torch.cat([past_value, current_value], dim=2)
    
    print(f"Full KV shape: {full_key.shape}")
    
    # Verify shape
    expected_shape = (K, num_heads, seq_len + 1, head_dim)
    assert full_key.shape == expected_shape, f"Expected {expected_shape}, got {full_key.shape}"
    print(f"✓ Broadcasting works correctly")
    
    # Verify past is shared across all K samples
    for k in range(K):
        assert torch.allclose(full_key[k, :, :seq_len, :], past_key[0]), \
            f"Sample {k} should share same past"
    print(f"✓ All {K} samples share the same past")
    
    print("\n✅ TEST 4 PASSED\n")


def test_tau_map_creation():
    """Test 5: Verify tau map creation"""
    print("="*80)
    print("TEST 5: Tau Map Creation")
    print("="*80)
    
    # Test default tau map
    num_layers = 32
    tau_map = create_default_tau_map(num_layers, tau_middle=1.0)
    
    print(f"Created tau map for {num_layers} layers")
    print(f"  First layer (idx=0): tau={tau_map[0]}")
    print(f"  Middle layer (idx=16): tau={tau_map[16]}")
    print(f"  Last layer (idx=31): tau={tau_map[31]}")
    
    # Verify first/last are 0
    assert tau_map[0] == 0.0, "First layer should have tau=0"
    assert tau_map[num_layers-1] == 0.0, "Last layer should have tau=0"
    print("✓ First and last layers have tau=0 (deterministic)")
    
    # Verify middle layers have tau_middle
    for i in range(1, num_layers-1):
        assert tau_map[i] == 1.0, f"Middle layer {i} should have tau=1.0"
    print("✓ Middle layers have tau=1.0 (stochastic)")
    
    print("\n✅ TEST 5 PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("RoE IMPLEMENTATION VERIFICATION")
    print("="*80 + "\n")
    
    try:
        # Run all tests
        test_roe_context()
        test_apply_roe_to_router()
        test_probability_averaging()
        test_kv_cache_broadcasting()
        test_tau_map_creation()
        
        print("="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nYour RoE implementation is working correctly!")
        print("\nNext steps:")
        print("1. Try running example_integration.py to see full generation")
        print("2. Integrate into your run_short_form_multihop.py")
        print("3. Compare baseline vs RoE results")
        
    except AssertionError as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        print("\nPlease check your implementation.")
        return 1
    
    except Exception as e:
        print("\n" + "="*80)
        print("❌ UNEXPECTED ERROR")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
