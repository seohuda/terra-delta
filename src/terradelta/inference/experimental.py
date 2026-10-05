"""Parsing of the optional ``experimental`` config block (v2.3.2). Everything defaults to OFF.

    experimental:
      pair_gate:
        enabled: false
      alignment_residual:
        enabled: false
        max_shift_image_px: 12
        bbox_padding: 8
        feature_scales: [2, 3]
        metric: l2

Configs without an ``experimental`` key keep the exact V2.3.1 behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .local_alignment import LocalAlignmentConfig
from .pair_gate import PairGateConfig


@dataclass(frozen=True)
class ExperimentalConfig:
    pair_gate: PairGateConfig = field(default_factory=PairGateConfig)
    alignment_residual: LocalAlignmentConfig = field(default_factory=LocalAlignmentConfig)

    @property
    def any_enabled(self) -> bool:
        return self.pair_gate.enabled or self.alignment_residual.enabled

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ExperimentalConfig:
        """Read ``config['experimental']``; absent or empty means everything disabled."""
        block = config.get("experimental")
        if block is None:
            return cls()
        if not isinstance(block, Mapping):
            raise ValueError("experimental config must be a mapping")
        unknown = set(block) - {"pair_gate", "alignment_residual"}
        if unknown:
            raise ValueError(f"Unknown experimental keys: {sorted(unknown)}")
        return cls(
            pair_gate=PairGateConfig.from_mapping(block.get("pair_gate")),
            alignment_residual=LocalAlignmentConfig.from_mapping(block.get("alignment_residual")),
        )
