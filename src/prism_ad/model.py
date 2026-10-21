"""Small, independent structural changes to the tested threshold PRISM-AD."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .threshold import ThresholdPrismAD, training_knots, DeterministicTrajectory
from .native import masked_softmax

VARIANTS = ("local_pairs", "relative_history", "combined")
DOMAIN_FEATURES = ("PHC_MEM_z", "PHC_EXF_z", "PHC_LAN_z", "PHC_VSP_z")


class LocalPairs(nn.Module):
    """Zero-anchored, 4x4 tensor-product functions for the same allowlist."""
    def __init__(self, pairs, training_static, names):
        super().__init__()
        edges, active = training_knots(training_static, names, bins=4)
        self.register_buffer("pairs", pairs.detach().cpu().clone())
        self.register_buffer("left", torch.tensor(edges[:, :-1]))
        self.register_buffer("width", torch.tensor(np.maximum(np.diff(edges), 1e-5)))
        self.register_buffer("active", torch.tensor(active, dtype=torch.float32))
        self.coefficient = nn.Parameter(torch.zeros(len(pairs), 4, 4))

    @property
    def n_pairs(self):
        return len(self.pairs)

    def basis(self, x):
        return ((x.unsqueeze(-1)-self.left)/self.width).clamp(0, 1)*self.active

    def forward(self, x):
        if not self.n_pairs:
            return x.new_zeros((len(x), 0))
        z = self.basis(x) - self.basis(torch.zeros_like(x))
        return torch.einsum("bpi,pij,bpj->bp", z[:, self.pairs[:, 0]],
                            self.coefficient, z[:, self.pairs[:, 1]])

    def group_penalty(self):
        c = self.coefficient
        if not self.n_pairs:
            return c.new_zeros(())
        a = self.active[self.pairs[:, 0], :, None]*self.active[self.pairs[:, 1], None, :]
        c = c*a
        return ((c.square().sum((1, 2))+1e-12).sqrt().sum()
                + (c[:, 1:]-c[:, :-1]).square().mean()
                + (c[:, :, 1:]-c[:, :, :-1]).square().mean())


class RelativeDomainHistory(DeterministicTrajectory):
    """One layer with within-domain attention and a fixed total risk bound."""
    def __init__(self, config, old):
        super().__init__(config)
        self.load_state_dict(old.state_dict())
        original = self.numeric_projection[0]
        expanded = nn.Linear(9, config.d_model)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :7].copy_(original.weight)
            expanded.bias.copy_(original.bias)
        self.numeric_projection[0] = expanded
        self.n_heads = config.n_heads

    def forward(self, numeric, domain, mask, severity):
        token = self.numeric_projection(numeric) + self.domain_embedding(domain)
        hidden = torch.cat((self.cls.expand(len(token), -1, -1), token), dim=1)
        padding = torch.cat((torch.zeros((len(mask), 1), dtype=torch.bool, device=mask.device), ~mask), dim=1)
        length = hidden.shape[1]
        allowed = torch.zeros((len(mask), length, length), dtype=torch.bool, device=mask.device)
        allowed[:, :, 0] = True  # Constant CLS key; no query can be fully masked.
        allowed[:, 1:, 1:] = domain[:, :, None] == domain[:, None, :]
        blocked = (~allowed).repeat_interleave(self.n_heads, dim=0)
        encoded = self.encoder(hidden, mask=blocked, src_key_padding_mask=padding)
        event_encoded = self.output_norm(encoded[:, 1:])
        scores = self.attention(event_encoded).squeeze(-1)
        # Four domain-specific normalizations, fixed weights: no cross-domain
        # count dilution and no missing-domain renormalization.
        attention = sum(masked_softmax(scores, mask & (domain == d)) for d in range(4))/4.
        value = self.value_bound*torch.tanh(self.value(event_encoded).squeeze(-1))
        gate = 1. + .5*torch.tanh(self.gate_slope*severity + self.gate_intercept)
        signed = attention*value*mask*gate[:, None]
        contributions = torch.stack([(signed*(domain == d)*mask).sum(1) for d in range(4)], dim=1)
        available = mask.any(1).to(signed.dtype)
        return dict(trajectory_logit=signed.sum(1),
                    trajectory_embedding=(attention[..., None]*event_encoded).sum(1)*available[:, None],
                    event_attention=attention, event_signed_contribution=signed,
                    domain_signed_contribution=contributions,
                    history_available=available, severity_gate=gate)


class StructuralPrismAD(ThresholdPrismAD):
    def __init__(self, names, training_static, variant, pre):
        if variant not in (*VARIANTS, "threshold"):
            raise ValueError("Unspecified structural candidate")
        super().__init__(names, training_static, "threshold")
        self.variant = variant
        if variant in ("local_pairs", "combined"):
            self.clinical_anchor.pairs = LocalPairs(self.clinical_anchor.pairs.pairs, training_static, names)
        self.relative_history = variant in ("relative_history", "combined")
        if self.relative_history:
            # Keep the parent's RNG position after initialization. Shared
            # initial parameters and stochastic training draws remain aligned.
            with torch.random.fork_rng(devices=[]):
                self.trajectory = RelativeDomainHistory(self.config, self.trajectory)
            transforms = {r.name: r for r in pre.numeric}
            self.register_buffer("current_index", torch.tensor([names.index(n) for n in DOMAIN_FEATURES]))
            self.register_buffer("current_missing_index", torch.tensor([names.index(n+"__missing") for n in DOMAIN_FEATURES]))
            self.register_buffer("current_scale", torch.tensor([transforms[n].scale for n in DOMAIN_FEATURES], dtype=torch.float32))
            self.register_buffer("current_median", torch.tensor([transforms[n].median for n in DOMAIN_FEATURES], dtype=torch.float32))
            self.register_buffer("event_mean", torch.tensor(pre.event_value_mean, dtype=torch.float32))
            self.register_buffer("event_scale", torch.tensor(pre.event_value_scale, dtype=torch.float32))
            if any(transforms[n].log1p for n in DOMAIN_FEATURES) or pre.event_value_scale <= 0:
                raise ValueError("Incompatible cognitive units")

    def relative_inputs(self, static, numeric, domain, mask):
        current_raw = static[:, self.current_index]*self.current_scale + self.current_median
        current_event_units = (current_raw-self.event_mean)/self.event_scale
        index = domain.clamp(0, 3)
        missing = (static[:, self.current_missing_index] > .5).gather(1, index)
        current = current_event_units.gather(1, index)
        contrast = torch.where(missing, torch.zeros_like(current), current-numeric[:, :, 0])
        extra = torch.stack((contrast*mask, (missing & mask).to(numeric.dtype)), dim=-1)
        return torch.cat((numeric, extra), dim=-1)

    def forward(self, static, prs, event_numeric, event_domain, event_mask, severity):
        if self.relative_history:
            event_numeric = self.relative_inputs(static, event_numeric, event_domain, event_mask)
        return super().forward(static, prs, event_numeric, event_domain, event_mask, severity)
