import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class ContextEncoder(nn.Module):
    """Encodes driving context to determine expert weights"""
    def __init__(self, context_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, context_dim] - driving context features
        Returns:
            context_features: [B, hidden_dim] - encoded context
        """
        return self.context_encoder(context)

class ExpertOutputProcessor(nn.Module):
    """Processes and normalizes expert outputs for gating"""
    def __init__(self, expert_output_dim: int, processed_dim: int = 256):
        super().__init__()
        self.expert_output_dim = expert_output_dim
        self.processed_dim = processed_dim
        
        self.processor = nn.Sequential(
            nn.Linear(expert_output_dim, processed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(processed_dim, processed_dim),
            nn.LayerNorm(processed_dim)
        )
        
    def forward(self, expert_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_output: [B, expert_output_dim] - raw expert output
        Returns:
            processed_output: [B, processed_dim] - normalized expert features
        """
        return self.processor(expert_output)


class GatingNetwork(nn.Module):
    """Mixture of Experts gating network"""
    def __init__(self, 
                 num_experts: int,
                 context_dim: int = 64,
                 expert_output_dims: List[int] = None,
                 processed_dim: int = 256,
                 hidden_dim: int = 128,
                 temperature: float = 1.0,
                 use_softmax: bool = True,
                 top_k: int = 0,
                 noise_type: str = 'gumbel',
                 noise_scale: float = 1.0,
                 apply_topk_at_eval: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.context_dim = context_dim
        self.processed_dim = processed_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.use_softmax = use_softmax
        # Noisy Top-K routing controls
        self.top_k = max(0, int(top_k))
        self.noise_type = noise_type
        self.noise_scale = float(noise_scale)
        self.apply_topk_at_eval = bool(apply_topk_at_eval)
        
        if expert_output_dims is None:
            expert_output_dims = [256] * num_experts
        
        self.context_encoder = ContextEncoder(context_dim, hidden_dim)
        
        self.expert_processors = nn.ModuleList([
            ExpertOutputProcessor(dim, processed_dim) 
            for dim in expert_output_dims
        ])
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim + processed_dim * num_experts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )
        
        self.output_projection = nn.Linear(processed_dim, processed_dim)

    def _sample_noise(self, shape, device):
        if self.noise_scale <= 0.0:
            return torch.zeros(shape, device=device)
        if self.noise_type.lower() == 'gumbel':
            # Gumbel(0,1): -log(-log(U))
            u = torch.rand(shape, device=device).clamp_(1e-6, 1 - 1e-6)
            return -torch.log(-torch.log(u)) * self.noise_scale
        elif self.noise_type.lower() == 'gaussian':
            return torch.randn(shape, device=device) * self.noise_scale
        else:
            return torch.zeros(shape, device=device)

    def _apply_topk_mask(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or k >= logits.size(1):
            return logits
        topk_vals, topk_idx = torch.topk(logits, k, dim=1)
        mask = torch.full_like(logits, fill_value=float('-inf'))
        mask.scatter_(1, topk_idx, topk_vals)
        return mask
        
    def forward(self, 
                expert_outputs: List[torch.Tensor], 
                context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            expert_outputs: List of [B, expert_output_dim] tensors from each expert
            context: [B, context_dim] - driving context features
        Returns:
            Dict containing:
                - combined_output: [B, processed_dim] - weighted combination of expert outputs
                - expert_weights: [B, num_experts] - learned weights for each expert
                - processed_expert_outputs: List of [B, processed_dim] - processed expert outputs
        """
        batch_size = context.size(0)
        
        context_features = self.context_encoder(context)  # [B, hidden_dim]
        
        processed_outputs = []
        for i, (expert_output, processor) in enumerate(zip(expert_outputs, self.expert_processors)):
            processed = processor(expert_output)  # [B, processed_dim]
            processed_outputs.append(processed)
        
        all_processed = torch.cat(processed_outputs, dim=1)  # [B, processed_dim * num_experts]
        
        gate_input = torch.cat([context_features, all_processed], dim=1)  # [B, hidden_dim + processed_dim * num_experts]
        gate_logits = self.gate_network(gate_input)  # [B, num_experts]

        apply_topk = (self.top_k > 0) and (self.training or self.apply_topk_at_eval)
        logits_for_softmax = gate_logits
        if apply_topk:
            noise = self._sample_noise(gate_logits.shape, gate_logits.device)
            noisy_logits = gate_logits + noise
            masked_logits = self._apply_topk_mask(noisy_logits, self.top_k)
            logits_for_softmax = masked_logits

        if self.use_softmax:
            gate_weights = F.softmax(logits_for_softmax / self.temperature, dim=1)  # [B, num_experts]
        else:
            gate_weights = torch.sigmoid(logits_for_softmax)  # [B, num_experts]
            gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        combined_output = torch.zeros(batch_size, self.processed_dim, device=context.device)
        for i, processed_output in enumerate(processed_outputs):
            weight = gate_weights[:, i:i+1]  # [B, 1]
            combined_output += weight * processed_output
        
        final_output = self.output_projection(combined_output)
        
        return {
            'combined_output': final_output,
            'expert_weights': gate_weights,
            'processed_expert_outputs': processed_outputs,
            'gate_logits': gate_logits
        }
    
    def get_expert_weights(self, context: torch.Tensor) -> torch.Tensor:
        """Get expert weights without processing expert outputs (for analysis)"""
        context_features = self.context_encoder(context)
        
        dummy_expert_outputs = [torch.zeros(context.size(0), self.processed_dim, device=context.device)] * self.num_experts
        all_processed = torch.cat(dummy_expert_outputs, dim=1)
        
        gate_input = torch.cat([context_features, all_processed], dim=1)
        gate_logits = self.gate_network(gate_input)

        # For analysis, do not apply noisy top-k unless explicitly set to apply at eval
        apply_topk = (self.top_k > 0) and self.apply_topk_at_eval
        logits_for_softmax = gate_logits
        if apply_topk:
            noise = self._sample_noise(gate_logits.shape, gate_logits.device)
            noisy_logits = gate_logits + noise
            logits_for_softmax = self._apply_topk_mask(noisy_logits, self.top_k)

        if self.use_softmax:
            return F.softmax(logits_for_softmax / self.temperature, dim=1)
        else:
            gate_weights = torch.sigmoid(logits_for_softmax)
            return gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)

    def get_gating_logits(self, context: torch.Tensor) -> torch.Tensor:
        """Return raw gating logits (no softmax), context-only path for analysis."""
        context_features = self.context_encoder(context)
        dummy_expert_outputs = [torch.zeros(context.size(0), self.processed_dim, device=context.device)] * self.num_experts
        all_processed = torch.cat(dummy_expert_outputs, dim=1)
        gate_input = torch.cat([context_features, all_processed], dim=1)
        return self.gate_network(gate_input)

class MoEArchitecture(nn.Module):
    """Complete Mixture of Experts architecture"""
    def __init__(self, 
                 experts: List[nn.Module],
                 gating_network: GatingNetwork,
                 policy_head: nn.Module):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_network = gating_network
        self.policy_head = policy_head
        
    def forward(self, 
                inputs: Dict[str, torch.Tensor], 
                context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: Dict containing inputs for each expert
            context: [B, context_dim] - driving context
        Returns:
            Dict containing policy outputs and gating information
        """
        # 1. Run all experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(inputs)  # Each expert handles its own input format
            expert_outputs.append(expert_output)
        
        # 2. Combine via gating network
        gating_output = self.gating_network(expert_outputs, context)
        
        # 3. Generate policy outputs
        policy_output = self.policy_head(gating_output['combined_output'])
        
        return {
            'policy_output': policy_output,
            'expert_weights': gating_output['expert_weights'],
            'combined_features': gating_output['combined_output'],
            'expert_outputs': expert_outputs
        }

