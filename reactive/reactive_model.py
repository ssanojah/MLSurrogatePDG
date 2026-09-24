"""
ReactiveTransformer: sliding-window Transformer guidance policy.

Maps a window of the last W physical states (B, W, 7), with an optional
padding mask, to one thrust command (B, 3).

Usage:
    model = ReactiveTransformer(ReactiveConfig(window_size=16))
    thrust = model(window, padding_mask=mask)   # (B, 3)

    python reactive_model.py             # run the built-in checks
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
class ReactiveConfig:
    """Hyperparameters for the reactive sliding-window Transformer."""

    # -- Problem dimensions --
    state_dim: int = 7       # [r_x, r_y, r_z, v_x, v_y, v_z, m]  (no t_f)
    action_dim: int = 3      # [T_cx, T_cy, T_cz] in MN

    # -- Sliding window --
    window_size: int = 16

    # -- Transformer architecture (same as model.py defaults) --
    d_model: int = 64        # embedding dimension
    nhead: int = 4           # attention heads (d_k = 64/4 = 16 per head)
    num_layers: int = 2      # encoder layers
    d_ff: int = 128          # feed-forward hidden dim (2 × d_model)
    dropout: float = 0.1     # dropout rate

    # -- Model type tag (for dispatch in shared scripts) --
    model_type: str = 'reactive_transformer'

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ReactiveTransformer(nn.Module):
    """Sliding-window causal Transformer for reactive PDG."""

    def __init__(self, config: Optional[ReactiveConfig] = None):
        super().__init__()

        if config is None:
            config = ReactiveConfig()
        self.config = config
        c = config

        # --- 1. Input Projection ---
        self.input_proj = nn.Linear(c.state_dim, c.d_model)

        # --- 2. Positional Encoding ---
        self.pos_embedding = nn.Embedding(c.window_size, c.d_model)

        # --- 3. Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.d_ff,
            dropout=c.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
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
        """Xavier init for weights, zeros for biases, scaled output head."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

        # Smaller init for output head (GPT-2 convention)
        with torch.no_grad():
            scale = 1.0 / math.sqrt(self.config.num_layers)
            self.output_head.weight.mul_(scale)

    def forward(
        self,
        window: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: window of recent states → single thrust prediction.

        Parameters
        ----------
        window       : (B, W, state_dim) normalized states
        padding_mask : (B, W) boolean, True = padding position to ignore.
                       None means no padding (all positions are real).

        Returns
        -------
        thrust : (B, action_dim) predicted normalized thrust at current step
        """
        B, W, _ = window.shape

        # 1. Input projection: (B, W, 7) → (B, W, d_model)
        x = self.input_proj(window)

        # 2. Add positional encoding
        positions = torch.arange(W, device=window.device)
        x = x + self.pos_embedding(positions)   # broadcasts over batch

        attn_mask = self._build_combined_mask(B, W, padding_mask, window.device)

        # 4. Transformer encoder — only `mask`, no `src_key_padding_mask`
        x = self.transformer_encoder(x, mask=attn_mask)
        # x shape: (B, W, d_model)

        x_current = x[:, -1, :]   # (B, d_model)

        # 6. Final norm + output projection
        x_current = self.final_norm(x_current)       # (B, d_model)
        thrust = self.output_head(x_current)          # (B, action_dim)

        return thrust

    def _build_combined_mask(
        self,
        B: int,
        W: int,
        padding_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Build a single float attention mask that encodes both causality and
        padding.
        """
        # Causal component: (W, W), -inf above diagonal
        causal = torch.triu(
            torch.ones(W, W, device=device) * float('-inf'),
            diagonal=1,
        )

        if padding_mask is None:
            return causal  # (W, W), no per-batch variation needed

        # Padding component: (B, 1, W), -inf where padded
        pad = torch.zeros(B, 1, W, device=device)
        pad.masked_fill_(padding_mask.unsqueeze(1), float('-inf'))

        # Combine: causal (W, W) broadcasts with pad (B, 1, W) → (B, W, W)
        combined = causal.unsqueeze(0) + pad   # (B, W, W)

        diag = torch.arange(W, device=device)
        combined[:, diag, diag] = 0.0

        # Expand for multi-head attention: (B, W, W) → (B*nhead, W, W)
        nhead = self.config.nhead
        combined = combined.unsqueeze(1).expand(B, nhead, W, W)
        combined = combined.reshape(B * nhead, W, W)

        return combined

    @staticmethod
    def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Float causal mask (used by _build_combined_mask internally). 0 where
        attention is allowed, -inf where blocked.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        c = self.config
        total_params = self.count_parameters()

        lines = [
            'ReactiveTransformer Architecture',
            '=' * 55,
            f'  State dim:      {c.state_dim}  (no t_f)',
            f'  Action dim:     {c.action_dim}',
            f'  Window size:    {c.window_size}',
            f'  d_model:        {c.d_model}',
            f'  Heads:          {c.nhead} (d_k = {c.d_model // c.nhead} per head)',
            f'  Layers:         {c.num_layers}',
            f'  FFN hidden dim: {c.d_ff}',
            f'  Dropout:        {c.dropout}',
            f'  Total params:   {total_params:,}',
            '',
            'Components:',
        ]
        for name, module in self.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            lines.append(f'  {name:.<30s} {n_params:>8,} params')

        # Comparison with the 60-node PDGTransformer
        lines.extend([
            '',
            'Comparison with PDGTransformer (model.py):',
            f'  PDGTransformer:     71,683 params  (60 seq, 8-dim state)',
            f'  ReactiveTransformer: {total_params:,} params  '
            f'({c.window_size} window, {c.state_dim}-dim state)',
        ])

        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------

def _verify():
    """Quick sanity checks — run with: python reactive_model.py"""
    import tempfile, os

    print("=" * 60)
    print("REACTIVE TRANSFORMER VERIFICATION")
    print("=" * 60)

    # --- 1. Instantiation ---
    config = ReactiveConfig(window_size=16)
    model = ReactiveTransformer(config)
    print(f"\n{model.summary()}\n")

    # --- 2. Forward pass shape ---
    B, W = 4, config.window_size
    dummy_window = torch.randn(B, W, config.state_dim)
    out = model(dummy_window)
    assert out.shape == (B, config.action_dim), \
        f"Output shape {out.shape}, expected ({B}, {config.action_dim})"
    print(f"  Forward pass shape: {out.shape} ✓")

    # --- 3. Forward with padding mask ---
    mask = torch.zeros(B, W, dtype=torch.bool)
    mask[:, :5] = True  # first 5 positions are padding
    out_masked = model(dummy_window, padding_mask=mask)
    assert out_masked.shape == (B, config.action_dim)
    print(f"  Forward with padding mask: {out_masked.shape} ✓")

    # Verify that padding changes the output (the model sees different data)
    assert not torch.allclose(out, out_masked), \
        "Padding mask had no effect — something is wrong"
    print(f"  Padding mask affects output ✓")

    # --- 4. Causal mask check ---
    causal = ReactiveTransformer._generate_causal_mask(W, torch.device('cpu'))
    assert causal.shape == (W, W)
    # Boolean: True above diagonal (blocked), False on/below (allowed)
    assert causal[0, 1] == float('-inf'), "Causal mask: future not blocked"
    assert causal[1, 0] == 0.0, "Causal mask: past blocked incorrectly"
    assert causal[0, 0] == 0.0, "Causal mask: self blocked"
    print(f"  Causal mask correct ✓")

    # --- 4b. Train vs eval consistency (NaN check in eval mode) ---
    model.train()
    out_train = model(dummy_window, padding_mask=mask)
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_window, padding_mask=mask)
    assert not torch.isnan(out_eval).any(), "NaN in eval mode — mask bug not fixed!"
    assert not torch.isnan(out_train).any(), "NaN in train mode"
    print(f"  Train/eval mode consistency (no NaN) ✓")

    # --- 5. Gradient flow ---
    target = torch.randn(B, config.action_dim)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters())
    assert n_grad == n_total, f"Only {n_grad}/{n_total} params got gradients"
    print(f"  Gradient flows to all {n_total} parameters ✓")

    # --- 6. Config roundtrip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'config.json')
        config.save(config_path)
        config_loaded = ReactiveConfig.load(config_path)
        assert config_loaded.window_size == config.window_size
        assert config_loaded.d_model == config.d_model
        assert config_loaded.state_dim == config.state_dim
        print(f"  Config save/load roundtrip ✓")

    # --- 7. Model save/load roundtrip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pt')
        torch.save(model.state_dict(), model_path)
        model2 = ReactiveTransformer(config)
        model2.load_state_dict(torch.load(model_path, weights_only=True))
        model2.eval()
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_window)
            out2 = model2(dummy_window)
        assert torch.allclose(out1, out2, atol=1e-6), "Model load mismatch"
        print(f"  Model save/load roundtrip ✓")

    # --- 8. Parameter count ---
    n_params = model.count_parameters()
    print(f"\n  Total parameters: {n_params:,}")
    assert 50_000 < n_params < 200_000, \
        f"Parameter count {n_params} seems off — check architecture"
    print(f"  Parameter count in expected range ✓")

    # --- 9. Single-sample inference ---
    model.eval()
    single_window = torch.randn(1, W, config.state_dim)
    with torch.no_grad():
        single_out = model(single_window)
    assert single_out.shape == (1, config.action_dim)
    print(f"  Single-sample inference: {single_out.shape} ✓")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("All checks passed ✓")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    _verify()
