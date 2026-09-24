"""
PDGMLP: feedforward MLP baseline policy.

Same interface as PDGTransformer, (B, N, 8) -> (B, N, 3), with each node
processed independently. Default: 6 hidden layers of 256, ReLU.

Usage:
    model = PDGMLP(MLPConfig())
    actions = model(states)              # (B, N, 3)

    model, cfg = load_model_from_config('model_config.json', 'model.pt')
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MLPConfig:
    """Hyperparameters for the MLP baseline policy."""
    # Problem dimensions (must match PDGTransformer / OCP)
    state_dim: int = 8       # [r_x, r_y, r_z, v_x, v_y, v_z, m, t_f]
    action_dim: int = 3      # [T_cx, T_cy, T_cz] in MN

    # MLP architecture (Li & Wang 2025 defaults)
    hidden_dim: int = 256    # neurons per hidden layer
    num_hidden_layers: int = 6   # number of hidden layers
    activation: str = 'relu'     # 'relu' (Li & Wang) or 'gelu'
    dropout: float = 0.0

    # Identifier for serialization (distinguishes from TransformerConfig)
    model_type: str = 'mlp'

    def save(self, filepath):
        """Save config to JSON for reproducibility."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load config from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Ignore 'model_type' if present but not in constructor
        # (forward-compatibility)
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PDGMLP(nn.Module):
    """Markovian feedforward policy for powered descent guidance.

    Parameters
    ----------
    config : MLPConfig
        Architecture hyperparameters.
    """

    def __init__(self, config: Optional[MLPConfig] = None):
        super().__init__()

        if config is None:
            config = MLPConfig()
        self.config = config

        c = config  # shorthand

        # Select activation function
        if c.activation == 'relu':
            act_fn = nn.ReLU
        elif c.activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {c.activation}")

        # Build the hidden layers as a Sequential stack
        layers = []
        in_dim = c.state_dim

        for i in range(c.num_hidden_layers):
            layers.append(nn.Linear(in_dim, c.hidden_dim))
            layers.append(act_fn())
            if c.dropout > 0:
                layers.append(nn.Dropout(c.dropout))
            in_dim = c.hidden_dim

        self.hidden = nn.Sequential(*layers)

        self.output_head = nn.Linear(c.hidden_dim, c.action_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for weight matrices, zeros for biases."""
        for name, p in self.named_parameters():
            if p.dim() > 1:  # weight matrices
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(
        self,
        states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: states → predicted actions.

        Parameters
        ----------
        states : (B, N, state_dim) tensor of normalized states
        mask   : IGNORED.  Accepted for interface compatibility with
                 PDGTransformer (which uses a causal mask).  The MLP
                 has no attention mechanism, so masking is meaningless.

        Returns
        -------
        actions : (B, N, action_dim) tensor of predicted (normalized)
                  actions
        """
        B, N, D = states.shape

        # Reshape: treat each (state, timestep) as an independent sample
        x = states.reshape(B * N, D)       # (B*N, 8)

        # Forward through hidden layers
        x = self.hidden(x)                 # (B*N, hidden_dim)

        # Output projection
        x = self.output_head(x)            # (B*N, 3)

        # Reshape back to sequence format
        actions = x.reshape(B, N, self.config.action_dim)  # (B, N, 3)

        return actions

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Human-readable model summary."""
        c = self.config
        total_params = self.count_parameters()

        lines = [
            'PDGMLP Architecture (Li & Wang 2025 baseline)',
            '=' * 55,
            f'  State dim:        {c.state_dim}',
            f'  Action dim:       {c.action_dim}',
            f'  Hidden dim:       {c.hidden_dim}',
            f'  Hidden layers:    {c.num_hidden_layers}',
            f'  Activation:       {c.activation}',
            f'  Dropout:          {c.dropout}',
            f'  Total params:     {total_params:,}',
            '',
            'Components:',
        ]

        # Hidden layers breakdown
        hidden_params = sum(
            p.numel() for p in self.hidden.parameters()
        )
        lines.append(f'  {"hidden":.<30s} {hidden_params:>8,} params')

        out_params = sum(
            p.numel() for p in self.output_head.parameters()
        )
        lines.append(f'  {"output_head":.<30s} {out_params:>8,} params')

        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Utility: load any model from a config JSON
# ---------------------------------------------------------------------------

def load_model_from_config(config_path, checkpoint_path=None, device='cpu'):
    """Load either a PDGMLP or PDGTransformer from a saved config JSON.

    Parameters
    ----------
    config_path      : path to model_config.json
    checkpoint_path  : path to .pt checkpoint (optional)
    device           : torch device

    Returns
    -------
    model : nn.Module (PDGMLP or PDGTransformer)
    config : MLPConfig or TransformerConfig
    """
    with open(config_path, 'r') as f:
        raw = json.load(f)

    model_type = raw.get('model_type', 'transformer')

    if model_type == 'mlp':
        config = MLPConfig.load(config_path)
        model = PDGMLP(config).to(device)
    else:
        # Import here to avoid circular dependency
        from model import TransformerConfig, PDGTransformer
        config = TransformerConfig.load(config_path)
        model = PDGTransformer(config).to(device)

    if checkpoint_path is not None:
        import torch
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

    return model, config
