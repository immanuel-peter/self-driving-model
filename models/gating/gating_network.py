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
        
        # Context features: speed, steering, throttle, brake, weather, time_of_day, etc.
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
                 use_softmax: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.context_dim = context_dim
        self.processed_dim = processed_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.use_softmax = use_softmax
        
        # Default expert output dimensions if not provided
        if expert_output_dims is None:
            expert_output_dims = [256] * num_experts  # Default to 256 for each expert
        
        # Context encoder
        self.context_encoder = ContextEncoder(context_dim, hidden_dim)
        
        # Expert output processors
        self.expert_processors = nn.ModuleList([
            ExpertOutputProcessor(dim, processed_dim) 
            for dim in expert_output_dims
        ])
        
        # Gating mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim + processed_dim * num_experts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # Output projection
        self.output_projection = nn.Linear(processed_dim, processed_dim)
        
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
        
        # 1. Encode context
        context_features = self.context_encoder(context)  # [B, hidden_dim]
        
        # 2. Process each expert output
        processed_outputs = []
        for i, (expert_output, processor) in enumerate(zip(expert_outputs, self.expert_processors)):
            processed = processor(expert_output)  # [B, processed_dim]
            processed_outputs.append(processed)
        
        # 3. Concatenate all processed outputs
        all_processed = torch.cat(processed_outputs, dim=1)  # [B, processed_dim * num_experts]
        
        # 4. Generate gating weights
        gate_input = torch.cat([context_features, all_processed], dim=1)  # [B, hidden_dim + processed_dim * num_experts]
        gate_logits = self.gate_network(gate_input)  # [B, num_experts]
        
        # 5. Apply temperature and softmax
        if self.use_softmax:
            gate_weights = F.softmax(gate_logits / self.temperature, dim=1)  # [B, num_experts]
        else:
            gate_weights = torch.sigmoid(gate_logits)  # [B, num_experts]
            # Normalize to sum to 1
            gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 6. Weighted combination of expert outputs
        combined_output = torch.zeros(batch_size, self.processed_dim, device=context.device)
        for i, processed_output in enumerate(processed_outputs):
            weight = gate_weights[:, i:i+1]  # [B, 1]
            combined_output += weight * processed_output
        
        # 7. Final projection
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
        
        # Use zeros for expert outputs to get context-only weights
        dummy_expert_outputs = [torch.zeros(context.size(0), self.processed_dim, device=context.device)] * self.num_experts
        all_processed = torch.cat(dummy_expert_outputs, dim=1)
        
        gate_input = torch.cat([context_features, all_processed], dim=1)
        gate_logits = self.gate_network(gate_input)
        
        if self.use_softmax:
            return F.softmax(gate_logits / self.temperature, dim=1)
        else:
            gate_weights = torch.sigmoid(gate_logits)
            return gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)

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

