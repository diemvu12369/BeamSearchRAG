"""
Modified OpenRAG MoE components with RoE support.

This file shows how to modify the OpenRAGGateAdapter to support RoE.
You need to integrate apply_roe_to_router() into your router's forward pass.
"""

import torch
import torch.nn as nn
import numpy as np
from openrag.configuration_openrag import OpenRAGConfig
from roe_openrag import apply_roe_to_router, get_roe_state


class ParallelAdapterMLPRoE(nn.Module):
    """Same as ParallelAdapterMLP but with RoE awareness"""
    def __init__(self, config, adapter_dim, adapter_scaling):
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.adapter_down = nn.Linear(self.hidden_size, adapter_dim, bias=False)
        self.adapter_up = nn.Linear(adapter_dim, self.hidden_size, bias=False)
        self.adapter_act = nn.GELU()
        self.adapter_dropout = nn.Dropout(p=0.1)
        self.adapter_scaling = adapter_scaling

    def forward(self, x):
        x = self.adapter_dropout(x)
        x = self.adapter_scaling * self.adapter_up(
            self.adapter_act(self.adapter_down(x))
        )
        return x


class OpenRAGGateAdapterRoE(nn.Module):
    """
    Modified OpenRAGGateAdapter with RoE support.
    
    Key changes:
    1. Apply RoE temperature to router logits before expert selection
    2. Handle batched inputs when K > 1
    """
    def __init__(self, config: OpenRAGConfig, layer_idx: int):
        super().__init__()
        
        self.layer_idx = layer_idx  # NEW: Track which layer this is
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        
        # Step 1: Router
        self.num_experts = config.num_experts
        self.topk = config.topk
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.dtype = getattr(torch, config.moe_dtype)
        
        # Step 2: Experts
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = ParallelAdapterMLPRoE(
                config, config.adapter_dim, config.moe_scaling
            )
    
    def forward(self, input_hidden_states, output_hidden_states, router_hidden_states):
        """
        Forward pass with RoE support.
        
        When RoE is active (K > 1):
        - input batch size = K * original_batch_size
        - Each of the K copies may select different experts due to tau > 0
        """
        orig_shape = output_hidden_states.shape
        input_hidden_states = input_hidden_states.view(-1, input_hidden_states.shape[-1])
        output_hidden_states = output_hidden_states.view(-1, output_hidden_states.shape[-1])
        router_hidden_states = router_hidden_states.view(-1, router_hidden_states.shape[-1])
        
        # Compute router logits
        router_logits = self.router(router_hidden_states)  # [batch, num_experts]
        
        # NEW: Apply RoE temperature to router logits
        # This is where the magic happens!
        # When tau > 0, Gumbel noise is added for stochastic expert selection
        router_logits = apply_roe_to_router(router_logits, self.layer_idx)
        
        # Select top-k experts (now potentially different across K samples due to RoE)
        expert_weights, expert_indices = torch.topk(router_logits, self.topk, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)
        flat_expert_indices = expert_indices.view(-1)
        
        # Replicate inputs for each selected expert
        input_hidden_states = input_hidden_states.repeat_interleave(self.topk, dim=0)
        expert_hidden_states = output_hidden_states.repeat_interleave(self.topk, dim=0)
        
        # Route to experts
        for idx, expert in enumerate(self.experts.values()):
            expert_mask = (flat_expert_indices == idx)
            if expert_mask.any():
                expert_hidden_states[expert_mask] += expert(
                    input_hidden_states[expert_mask]
                )
        
        # Aggregate expert outputs
        hidden_states = (
            expert_hidden_states.view(*expert_weights.shape, -1)
            * expert_weights.unsqueeze(-1)
        ).sum(dim=1)
        
        return hidden_states.view(*orig_shape), router_logits


class LlamaMLPRoE(nn.Module):
    """
    Modified LlamaMLP with RoE-enabled MoE adapter.
    
    This is a drop-in replacement for the original LlamaMLP.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Use appropriate activation function
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]
        
        # NEW: Pass layer_idx to MoE adapter
        self.moe_adapter = OpenRAGGateAdapterRoE(config, layer_idx)
    
    def forward(self, x):
        router_hidden_states = x
        up_proj = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(up_proj)
        
        # Apply MoE adapter with RoE support
        down_proj, router_logits = self.moe_adapter(
            down_proj, down_proj, router_hidden_states
        )
        
        return down_proj, router_logits
