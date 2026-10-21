"""Independent, explicitly additive local-threshold PRISM-AD candidates."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .constants import PAIR_ALLOWLIST
from .native import PrismAD, PrismADConfig, PurePairInteractionBank, OneLayerTrajectoryEncoder, masked_softmax


PROFILE_PAIRS = (
    ("global_z", "PHC_LAN_z"),
    ("global_z", "PHC_VSP_z"),
    ("PHC_MEM_z", "PHC_EXF_z"),
    ("PHC_MEM_z", "PHC_LAN_z"),
    ("PHC_EXF_z", "PHC_LAN_z"),
    ("PHC_LAN_z", "PHC_VSP_z"),
    ("PHC_LAN_z", "strict_history_language_last"),
    ("PHC_LAN_z", "strict_history_language_pace_per_5y"),
    ("PHC_VSP_z", "strict_history_visuospatial_last"),
    ("PHC_VSP_z", "strict_history_visuospatial_pace_per_5y"),
    ("PHC_Education", "PHC_MEM_z"),
    ("visit_age", "PHC_LAN_z"),
)


class DeterministicTrajectory(OneLayerTrajectoryEncoder):
    """Same encoder; explicit four-domain sums replace unsupported CUDA scatter.

    PyTorch 1.12 cannot deterministically scatter-add on CUDA. This changes only
    the computation of an explanatory output, not the event or risk functions.
    """
    def forward(self, numeric, domain, mask, severity):
        token = self.numeric_projection(numeric) + self.domain_embedding(domain)
        hidden = torch.cat((self.cls.expand(token.shape[0], -1, -1), token), dim=1)
        padding = torch.cat((torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device), ~mask), dim=1)
        encoded = self.encoder(hidden, src_key_padding_mask=padding)
        event_encoded = self.output_norm(encoded[:, 1:])
        scores = self.attention(event_encoded).squeeze(-1)
        attention = masked_softmax(scores, mask)
        event_value = self.value_bound * torch.tanh(self.value(event_encoded).squeeze(-1))
        ungated = attention * event_value * mask.to(event_value.dtype)
        severity_gate = 1. + .5 * torch.tanh(self.gate_slope * severity + self.gate_intercept)
        signed = ungated * severity_gate.unsqueeze(-1)
        available = mask.any(dim=1).to(signed.dtype)
        trajectory = signed.sum(dim=1) * available
        embedding = (attention.unsqueeze(-1) * event_encoded).sum(dim=1) * available.unsqueeze(-1)
        domain_contribution = torch.stack([(signed * (domain == d).to(signed.dtype) * mask).sum(1) for d in range(4)], dim=1)
        return dict(trajectory_logit=trajectory, trajectory_embedding=embedding,
                    event_attention=attention, event_signed_contribution=signed,
                    domain_signed_contribution=domain_contribution,
                    history_available=available, severity_gate=severity_gate)


def training_knots(values, names, bins=8):
    """Fit only training values, excluding imputed numeric observations."""
    n = values.shape[1]
    edges = np.zeros((n, bins + 1), dtype=np.float32)
    active = np.zeros((n, bins), dtype=bool)
    name_index = {name: j for j, name in enumerate(names)}
    for j, name in enumerate(names):
        if "=" in name or name.endswith("__missing"):
            continue
        observed = np.isfinite(values[:, j])
        if name + "__missing" in name_index:
            observed &= values[:, name_index[name + "__missing"]] < .5
        x = values[observed, j]
        if len(np.unique(x)) < 10:
            continue
        edges[j] = np.quantile(x, np.linspace(0, 1, bins + 1))
        active[j] = np.diff(edges[j]) > 1e-5
    return edges, active


class ThresholdMain(nn.Module):
    """Original featurewise network plus centered piecewise-linear curves."""
    def __init__(self, base, edges, active):
        super().__init__()
        self.base = base
        self.register_buffer("left", torch.tensor(edges[:, :-1]))
        self.register_buffer("width", torch.tensor(np.maximum(np.diff(edges), 1e-5)))
        self.register_buffer("active", torch.tensor(active, dtype=torch.float32))
        self.coefficient = nn.Parameter(torch.zeros_like(self.left))

    def basis(self, x):
        return ((x.unsqueeze(-1) - self.left) / self.width).clamp(0, 1) * self.active

    def forward(self, x):
        basis = self.basis(x) - self.basis(torch.zeros_like(x))
        return self.base(x) + (basis * self.coefficient).sum(-1)

    def penalty(self):
        c = self.coefficient * self.active
        return c.square().mean() + (c[:, 1:] - c[:, :-1]).square().mean()


class ThresholdPrismAD(PrismAD):
    def __init__(self, names, training_static, variant):
        if variant not in ("reference", "threshold", "threshold_profile"):
            raise ValueError("Unspecified candidate")
        super().__init__(PrismADConfig(name="interpretable_anchor"), names)
        trajectory = DeterministicTrajectory(self.config)
        trajectory.load_state_dict(self.trajectory.state_dict())
        self.trajectory = trajectory
        self.variant = variant
        if variant != "reference":
            edges, active = training_knots(training_static, names)
            self.clinical_anchor.main = ThresholdMain(self.clinical_anchor.main, edges, active)
        if variant == "threshold_profile":
            index = {name: j for j, name in enumerate(names)}
            self.pair_names = tuple(pair for pair in (*PAIR_ALLOWLIST, *PROFILE_PAIRS)
                                    if all(name in index for name in pair))
            pairs = tuple((index[a], index[b]) for a, b in self.pair_names)
            bank = PurePairInteractionBank(pairs, self.config.pair_hidden)
            old = self.clinical_anchor.pairs
            # Identical initial old-pair functions, plus zero-output new pairs.
            with torch.no_grad():
                for field in ("input_weight", "input_bias", "output_weight"):
                    getattr(bank, field)[:old.n_pairs].copy_(getattr(old, field))
                bank.output_weight[old.n_pairs:].zero_()
            self.clinical_anchor.pairs = bank

    def regularization_penalty(self):
        penalty = super().regularization_penalty()
        if isinstance(self.clinical_anchor.main, ThresholdMain):
            penalty = penalty + self.clinical_anchor.main.penalty()
        return penalty
