"""Low-capacity PRISM-AD: NAM/GA2M anchor plus one-layer past encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .constants import EVENT_NUMERIC_FEATURES, PAIR_ALLOWLIST, TRACKS


def masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """Softmax over valid events with an exact all-masked result of zero."""
    masked = scores.masked_fill(~mask, -1.0e4)
    weights = torch.softmax(masked, dim=-1) * mask.to(scores.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


@dataclass(frozen=True)
class PrismADConfig:
    name: str
    track: str = "D"
    main_hidden: int = 16
    pair_hidden: int = 8
    residual_width: int = 0
    residual_bound: float = 0.75
    d_model: int = 32
    n_heads: int = 4
    ff_dim: int = 64
    dropout: float = 0.10
    trajectory_value_bound: float = 0.75

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise ValueError("Unknown information track")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.main_hidden, self.pair_hidden, self.d_model, self.ff_dim) < 1:
            raise ValueError("Model widths must be positive")
        if self.residual_width < 0 or self.residual_bound <= 0 or self.trajectory_value_bound <= 0:
            raise ValueError("Invalid bounded-component configuration")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FeaturewiseNAM(nn.Module):
    """Independent nonlinear shape function for every transformed feature."""

    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.input_weight = nn.Parameter(torch.empty(n_features, hidden))
        self.input_bias = nn.Parameter(torch.zeros(n_features, hidden))
        self.output_weight = nn.Parameter(torch.empty(n_features, hidden))
        self.linear_weight = nn.Parameter(torch.zeros(n_features))
        nn.init.xavier_uniform_(self.input_weight)
        nn.init.normal_(self.output_weight, std=0.025)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError("Static matrix has the wrong width")
        expanded = values.unsqueeze(-1)
        hidden = F.silu(expanded * self.input_weight + self.input_bias)
        reference = F.silu(self.input_bias).unsqueeze(0)
        nonlinear = ((hidden - reference) * self.output_weight).sum(dim=-1)
        return nonlinear + values * self.linear_weight.unsqueeze(0)


class PurePairInteractionBank(nn.Module):
    """Allowlisted two-variable functions with pure GA2M decomposition.

    For pair f(x,y), the reported interaction is
    f(x,y)-f(x,0)-f(0,y)+f(0,0), so it is exactly zero when either endpoint is
    at its training-fold reference value.
    """

    def __init__(self, pairs: Sequence[tuple[int, int]], hidden: int) -> None:
        super().__init__()
        pair_tensor = (
            torch.tensor(tuple(pairs), dtype=torch.long)
            if pairs
            else torch.empty((0, 2), dtype=torch.long)
        )
        self.register_buffer("pairs", pair_tensor)
        n_pairs = int(pair_tensor.shape[0])
        self.input_weight = nn.Parameter(torch.empty(n_pairs, 2, hidden))
        self.input_bias = nn.Parameter(torch.zeros(n_pairs, hidden))
        self.output_weight = nn.Parameter(torch.empty(n_pairs, hidden))
        if n_pairs:
            nn.init.xavier_uniform_(self.input_weight)
            nn.init.normal_(self.output_weight, std=0.02)

    @property
    def n_pairs(self) -> int:
        return int(self.pairs.shape[0])

    def _evaluate(self, pair_values: Tensor) -> Tensor:
        hidden = torch.tanh(
            torch.einsum("bpj,pjh->bph", pair_values, self.input_weight)
            + self.input_bias.unsqueeze(0)
        )
        return (hidden * self.output_weight.unsqueeze(0)).sum(dim=-1)

    def forward(self, values: Tensor) -> Tensor:
        if self.n_pairs == 0:
            return values.new_zeros((values.shape[0], 0))
        pair_values = values[:, self.pairs]
        zeros = torch.zeros_like(pair_values)
        left_only = torch.stack(
            (pair_values[..., 0], zeros[..., 1]), dim=-1
        )
        right_only = torch.stack(
            (zeros[..., 0], pair_values[..., 1]), dim=-1
        )
        return (
            self._evaluate(pair_values)
            - self._evaluate(left_only)
            - self._evaluate(right_only)
            + self._evaluate(zeros)
        )

    def group_penalty(self) -> Tensor:
        if self.n_pairs == 0:
            return self.output_weight.new_zeros(())
        return torch.sqrt(self.output_weight.pow(2).sum(dim=1) + 1.0e-12).sum()


class BoundedClinicalResidual(nn.Module):
    """Optional compact residual, separated and bounded in the risk ledger."""

    def __init__(self, n_features: int, width: int, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.input = nn.Linear(n_features, width)
        self.output = nn.Linear(width, 1)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: Tensor) -> Tensor:
        raw = self.output(F.gelu(self.input(values))).squeeze(-1)
        reference = self.output(F.gelu(self.input(torch.zeros_like(values)))).squeeze(-1)
        return self.bound * torch.tanh(raw - reference)


class ClinicalAnchor(nn.Module):
    def __init__(
        self,
        n_features: int,
        pair_indices: Sequence[tuple[int, int]],
        config: PrismADConfig,
    ) -> None:
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(()))
        self.main = FeaturewiseNAM(n_features, config.main_hidden)
        self.pairs = PurePairInteractionBank(pair_indices, config.pair_hidden)
        self.residual = (
            BoundedClinicalResidual(
                n_features, config.residual_width, config.residual_bound
            )
            if config.residual_width > 0
            else None
        )

    def forward(self, values: Tensor) -> dict[str, Tensor]:
        main = self.main(values)
        pairs = self.pairs(values)
        residual = (
            self.residual(values)
            if self.residual is not None
            else values.new_zeros(values.shape[0])
        )
        intercept = self.intercept.expand(values.shape[0])
        main_logit = main.sum(dim=1)
        pair_logit = pairs.sum(dim=1)
        anchor = intercept + main_logit + pair_logit + residual
        return {
            "clinical_intercept": intercept,
            "clinical_main_contribution": main,
            "clinical_pair_contribution": pairs,
            "clinical_main_logit": main_logit,
            "clinical_pair_logit": pair_logit,
            "clinical_residual_logit": residual,
            "anchor_logit": anchor,
        }


class OneLayerTrajectoryEncoder(nn.Module):
    """One Transformer layer over events that all precede the landmark."""

    def __init__(self, config: PrismADConfig) -> None:
        super().__init__()
        self.value_bound = float(config.trajectory_value_bound)
        self.domain_embedding = nn.Embedding(5, config.d_model, padding_idx=4)
        self.numeric_projection = nn.Sequential(
            nn.Linear(len(EVENT_NUMERIC_FEATURES), config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(config.d_model)
        attention_width = max(8, config.d_model // 2)
        self.attention = nn.Sequential(
            nn.Linear(config.d_model, attention_width),
            nn.Tanh(),
            nn.Linear(attention_width, 1, bias=False),
        )
        self.value = nn.Sequential(
            nn.Linear(config.d_model, attention_width),
            nn.GELU(),
            nn.Linear(attention_width, 1),
        )
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)
        self.gate_slope = nn.Parameter(torch.zeros(()))
        self.gate_intercept = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.cls, std=0.02)

    def forward(
        self,
        numeric: Tensor,
        domain: Tensor,
        mask: Tensor,
        severity: Tensor,
    ) -> dict[str, Tensor]:
        token = self.numeric_projection(numeric) + self.domain_embedding(domain)
        cls = self.cls.expand(token.shape[0], -1, -1)
        hidden = torch.cat((cls, token), dim=1)
        padding = torch.cat(
            (
                torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device),
                ~mask,
            ),
            dim=1,
        )
        encoded = self.encoder(hidden, src_key_padding_mask=padding)
        event_encoded = self.output_norm(encoded[:, 1:])
        scores = self.attention(event_encoded).squeeze(-1)
        attention = masked_softmax(scores, mask)
        event_value = self.value_bound * torch.tanh(self.value(event_encoded).squeeze(-1))
        ungated = attention * event_value * mask.to(event_value.dtype)
        severity_gate = 1.0 + 0.5 * torch.tanh(
            self.gate_slope * severity + self.gate_intercept
        )
        signed = ungated * severity_gate.unsqueeze(-1)
        available = mask.any(dim=1).to(signed.dtype)
        trajectory = signed.sum(dim=1) * available
        embedding = (attention.unsqueeze(-1) * event_encoded).sum(dim=1)
        embedding = embedding * available.unsqueeze(-1)
        domain_contribution = torch.zeros(
            (domain.shape[0], 4), dtype=signed.dtype, device=signed.device
        )
        domain_contribution.scatter_add_(
            1,
            domain.clamp(min=0, max=3),
            signed * mask.to(signed.dtype),
        )
        return {
            "trajectory_logit": trajectory,
            "trajectory_embedding": embedding,
            "event_attention": attention,
            "event_signed_contribution": signed,
            "domain_signed_contribution": domain_contribution,
            "history_available": available,
            "severity_gate": severity_gate,
        }


class PrismAD(nn.Module):
    """Independent PRISM-AD with an exact additive branch-level risk ledger."""

    def __init__(
        self,
        config: PrismADConfig,
        static_feature_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.config = config
        self.static_feature_names = tuple(static_feature_names)
        feature_index = {
            name: index for index, name in enumerate(self.static_feature_names)
        }
        retained_pairs = tuple(
            (feature_index[left], feature_index[right])
            for left, right in PAIR_ALLOWLIST
            if left in feature_index and right in feature_index
        )
        self.pair_names = tuple(
            (left, right)
            for left, right in PAIR_ALLOWLIST
            if left in feature_index and right in feature_index
        )
        self.clinical_anchor = ClinicalAnchor(
            len(self.static_feature_names), retained_pairs, config
        )
        spec = TRACKS[config.track]
        self.use_prs = bool(spec.use_prs)
        self.use_sequence = bool(spec.use_sequence)
        if self.use_prs:
            self.prs_raw_coefficient = nn.Parameter(torch.tensor(-2.25))
        else:
            self.register_parameter("prs_raw_coefficient", None)
        self.trajectory = OneLayerTrajectoryEncoder(config) if self.use_sequence else None

    def effective_prs_coefficient(self) -> Tensor:
        if self.prs_raw_coefficient is None:
            return self.clinical_anchor.intercept.new_zeros(())
        return F.softplus(self.prs_raw_coefficient)

    def forward(
        self,
        static: Tensor,
        prs: Tensor,
        event_numeric: Tensor,
        event_domain: Tensor,
        event_mask: Tensor,
        severity: Tensor,
    ) -> dict[str, Tensor]:
        clinical = self.clinical_anchor(static)
        prs_logit = (
            self.effective_prs_coefficient() * prs.squeeze(-1)
            if self.use_prs
            else static.new_zeros(static.shape[0])
        )
        if self.trajectory is None:
            trajectory = {
                "trajectory_logit": static.new_zeros(static.shape[0]),
                "trajectory_embedding": static.new_zeros((static.shape[0], 0)),
                "event_attention": static.new_zeros(event_mask.shape),
                "event_signed_contribution": static.new_zeros(event_mask.shape),
                "domain_signed_contribution": static.new_zeros((static.shape[0], 4)),
                "history_available": static.new_zeros(static.shape[0]),
                "severity_gate": static.new_ones(static.shape[0]),
            }
        else:
            trajectory = self.trajectory(
                event_numeric, event_domain, event_mask, severity
            )
        risk = clinical["anchor_logit"] + prs_logit + trajectory["trajectory_logit"]
        ledger = torch.stack(
            (
                clinical["clinical_intercept"],
                clinical["clinical_main_logit"],
                clinical["clinical_pair_logit"],
                clinical["clinical_residual_logit"],
                prs_logit,
                trajectory["trajectory_logit"],
            ),
            dim=1,
        )
        return {
            **clinical,
            **trajectory,
            "prs_logit": prs_logit,
            "risk_logit": risk,
            "risk_group_contribution": ledger,
        }

    def regularization_penalty(self) -> Tensor:
        return self.clinical_anchor.pairs.group_penalty()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

