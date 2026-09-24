"""
PDGTransformer: causal Transformer guidance policy.

Maps a sequence of normalised states (B, N, 8) to normalised thrust
commands (B, N, 3). Default: d_model 64, 4 heads, 2 layers, d_ff 128.

Usage:
    model = PDGTransformer(TransformerConfig())
    actions = model(states)              # (B, N, 3)
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Hyperparameters for the PDG Transformer."""
    # Problem dimensions (match the OCP)
    state_dim: int = 8       # [r_x, r_y, r_z, v_x, v_y, v_z, m, t_f]
    action_dim: int = 3      # [T_cx, T_cy, T_cz] in MN

    # Transformer architecture
    d_model: int = 64        # embedding dimension
    nhead: int = 4           # number of attention heads (d_model / nhead = 16 per head)
    num_layers: int = 2      # number of transformer encoder layers
    d_ff: int = 128          # feed-forward hidden dimension (2 × d_model)
    dropout: float = 0.1     # dropout rate
    max_seq_len: int = 60    # maximum sequence length (= N control intervals)

    def save(self, filepath):
        """Save config to JSON for reproducibility."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load config from JSON."""
        with open(filepath, 'r') as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PDGTransformer(nn.Module):
    """Causal transformer policy for powered descent guidance.

    Parameters
    ----------
    config : TransformerConfig
        Architecture hyperparameters.
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()

        if config is None:
            config = TransformerConfig()
        self.config = config

        c = config  # shorthand

        # --- 1. Input Projection ---
        self.input_proj = nn.Linear(c.state_dim, c.d_model)

        # --- 2. Positional Encoding ---
        self.pos_embedding = nn.Embedding(c.max_seq_len, c.d_model)

        # --- 3. Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.d_ff,
            dropout=c.dropout,
            activation='gelu',
            batch_first=True,       # input shape (batch, seq, feature)
            norm_first=True,        # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=c.num_layers,
        )

        # --- 4. Final LayerNorm ---
        self.final_norm = nn.LayerNorm(c.d_model)

        # --- 5. Output Head ---
        self.output_head = nn.Linear(c.d_model, c.action_dim)

        # --- Initialize weights ---
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for weight matrices, zeros for biases."""
        for name, p in self.named_parameters():
            if p.dim() > 1:  # weight matrices
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

        # Smaller init for output projection (residual path scaling)
        with torch.no_grad():
            scale = 1.0 / math.sqrt(self.config.num_layers)
            self.output_head.weight.mul_(scale)

    def forward(
        self,
        states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: states → predicted actions.

        Parameters
        ----------
        states : (B, N, state_dim) tensor of normalized states
        mask   : optional (N, N) causal mask. If None, generates one
                 automatically. Pass explicitly if you want to reuse
                 a pre-computed mask across batches.

        Returns
        -------
        actions : (B, N, action_dim) tensor of predicted (normalized) actions
        """
        B, N, _ = states.shape

        # 1. Input projection: (B, N, state_dim) → (B, N, d_model)
        x = self.input_proj(states)

        # 2. Add positional encoding
        positions = torch.arange(N, device=states.device)  # (N,)
        x = x + self.pos_embedding(positions)               # broadcast over batch

        # 3. Generate causal mask if not provided
        if mask is None:
            mask = self._generate_causal_mask(N, states.device)

        # 4. Transformer encoder with causal masking
        x = self.transformer_encoder(x, mask=mask)  # (B, N, d_model)

        # 5. Final norm + output projection
        x = self.final_norm(x)                       # (B, N, d_model)
        actions = self.output_head(x)                # (B, N, action_dim)

        return actions

    @staticmethod
    def _generate_causal_mask(
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate an additive causal mask for the transformer."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )
        return mask

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Human-readable model summary."""
        c = self.config
        total_params = self.count_parameters()

        lines = [
            'PDGTransformer Architecture',
            '=' * 55,
            f'  State dim:      {c.state_dim}',
            f'  Action dim:     {c.action_dim}',
            f'  d_model:        {c.d_model}',
            f'  Heads:          {c.nhead} (d_k = {c.d_model // c.nhead} per head)',
            f'  Layers:         {c.num_layers}',
            f'  FFN hidden dim: {c.d_ff}',
            f'  Max seq len:    {c.max_seq_len}',
            f'  Dropout:        {c.dropout}',
            f'  Total params:   {total_params:,}',
            '',
            'Components:',
        ]

        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            lines.append(f'  {name:.<30s} {n_params:>8,} params')

        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class BCLoss(nn.Module):
    """Behavioural cloning loss: MSE between predicted and expert actions."""

    def __init__(
        self,
        component_weights: Optional[torch.Tensor] = None,
        timestep_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if component_weights is not None:
            self.register_buffer("component_weights", component_weights)
        else:
            self.component_weights = None

        if timestep_weights is not None:
            self.register_buffer("timestep_weights", timestep_weights)
        else:
            self.timestep_weights = None

    def forward(
        self,
        pred_actions: torch.Tensor,   # (B, N, 3)
        expert_actions: torch.Tensor,  # (B, N, 3)
    ) -> torch.Tensor:
        """Compute mean squared error over all timesteps and components."""
        diff = pred_actions - expert_actions
        sq_err = diff**2

        if self.component_weights is not None:
            sq_err = sq_err * self.component_weights

        if self.timestep_weights is not None:
            sq_err = sq_err * self.timestep_weights.view(1, -1, 1)

        return sq_err.mean()