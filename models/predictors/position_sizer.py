"""
Position Sizer: Continuous Probability-Based Position Sizing

Uses GNN regime probabilities to calculate optimal position size.
Formula: position = p(RISK_ON)*w_ro + p(CAUTION)*w_ca + p(RISK_OFF)*w_rf

Optimized weights from Phase 5 grid search:
- RISK_ON: 0.85
- CAUTION: 0.65
- RISK_OFF: 0.30

Performance (validation period):
- Max DD: 23.9% (target <25%) ✓
- Return: +19.4%
- Sharpe: 0.54

Author: Claude Opus 4.5
Date: December 2024
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ContinuousPositionSizer:
    """
    Calculate position size based on GNN regime probabilities.

    Uses weighted average of regime probabilities to produce
    smooth, continuous position sizing.
    """

    # Default weights from Phase 5 optimization
    DEFAULT_WEIGHTS = {
        'risk_on': 0.85,
        'caution': 0.65,
        'risk_off': 0.30
    }

    def __init__(
        self,
        risk_on_weight: float = None,
        caution_weight: float = None,
        risk_off_weight: float = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize position sizer.

        Args:
            risk_on_weight: Weight for RISK_ON regime (default 0.85)
            caution_weight: Weight for CAUTION regime (default 0.65)
            risk_off_weight: Weight for RISK_OFF regime (default 0.30)
            config_path: Path to JSON config file (overrides individual weights)
        """
        # Load from config file if provided
        if config_path and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.risk_on_weight = config.get('risk_on_weight', self.DEFAULT_WEIGHTS['risk_on'])
            self.caution_weight = config.get('caution_weight', self.DEFAULT_WEIGHTS['caution'])
            self.risk_off_weight = config.get('risk_off_weight', self.DEFAULT_WEIGHTS['risk_off'])
            logger.info(f"Loaded position sizing config from {config_path}")
        else:
            # Use provided weights or defaults
            self.risk_on_weight = risk_on_weight or self.DEFAULT_WEIGHTS['risk_on']
            self.caution_weight = caution_weight or self.DEFAULT_WEIGHTS['caution']
            self.risk_off_weight = risk_off_weight or self.DEFAULT_WEIGHTS['risk_off']

        logger.info(
            f"Position sizer initialized: RISK_ON={self.risk_on_weight:.0%}, "
            f"CAUTION={self.caution_weight:.0%}, RISK_OFF={self.risk_off_weight:.0%}"
        )

    def calculate_position(self, regime_probs: np.ndarray) -> float:
        """
        Calculate position size from regime probabilities.

        Args:
            regime_probs: Array of [p(RISK_ON), p(CAUTION), p(RISK_OFF)]

        Returns:
            Position size between 0 and 1
        """
        if len(regime_probs) != 3:
            raise ValueError(f"Expected 3 probabilities, got {len(regime_probs)}")

        position = (
            regime_probs[0] * self.risk_on_weight +
            regime_probs[1] * self.caution_weight +
            regime_probs[2] * self.risk_off_weight
        )

        # Clamp to valid range
        return float(np.clip(position, 0.0, 1.0))

    def calculate_positions_batch(self, regime_probs: np.ndarray) -> np.ndarray:
        """
        Calculate position sizes for multiple timesteps.

        Args:
            regime_probs: Array of shape (N, 3) with regime probabilities

        Returns:
            Array of position sizes
        """
        positions = (
            regime_probs[:, 0] * self.risk_on_weight +
            regime_probs[:, 1] * self.caution_weight +
            regime_probs[:, 2] * self.risk_off_weight
        )

        return np.clip(positions, 0.0, 1.0)

    def get_regime_label(self, regime_probs: np.ndarray) -> Tuple[str, float]:
        """
        Get dominant regime label and confidence.

        Args:
            regime_probs: Array of [p(RISK_ON), p(CAUTION), p(RISK_OFF)]

        Returns:
            Tuple of (regime_label, confidence)
        """
        labels = ['RISK_ON', 'CAUTION', 'RISK_OFF']
        idx = np.argmax(regime_probs)
        return labels[idx], float(regime_probs[idx])

    def get_config(self) -> Dict[str, float]:
        """Return current configuration."""
        return {
            'risk_on_weight': self.risk_on_weight,
            'caution_weight': self.caution_weight,
            'risk_off_weight': self.risk_off_weight
        }


class FixedRulesPositionSizer:
    """
    Original fixed rules position sizer (for comparison).

    Uses hard thresholds:
    - RISK_ON → 100%
    - CAUTION → 50%
    - RISK_OFF → 20%
    """

    POSITIONS = {
        'RISK_ON': 1.00,
        'CAUTION': 0.50,
        'RISK_OFF': 0.20
    }

    def calculate_position(self, regime_probs: np.ndarray) -> float:
        """Calculate position based on dominant regime."""
        labels = ['RISK_ON', 'CAUTION', 'RISK_OFF']
        idx = np.argmax(regime_probs)
        return self.POSITIONS[labels[idx]]

    def calculate_positions_batch(self, regime_probs: np.ndarray) -> np.ndarray:
        """Calculate positions for batch."""
        positions = []
        for probs in regime_probs:
            positions.append(self.calculate_position(probs))
        return np.array(positions)


# Factory function
def create_position_sizer(
    method: str = 'continuous',
    config_path: Optional[Path] = None
) -> ContinuousPositionSizer:
    """
    Create a position sizer.

    Args:
        method: 'continuous' or 'fixed'
        config_path: Path to config file

    Returns:
        Position sizer instance
    """
    if method == 'fixed':
        return FixedRulesPositionSizer()

    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'position_sizing.json'

    return ContinuousPositionSizer(config_path=config_path)
