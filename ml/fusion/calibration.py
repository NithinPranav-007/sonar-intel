"""
Confidence Calibration Module for SONAR-INTEL.

Transforms raw fused multi-modal scores into calibrated probabilistic confidence:
P(True Positive | Fused Score).

Provides:
1. BaseCalibrator (abstract interface)
2. IdentityCalibrator (passthrough baseline)
3. TemperatureCalibrator (temperature scaling for logit calibration)
4. PlattCalibrator (logistic calibration with versioned parameterization)

Parameters are strictly versioned to preserve empirical provenance.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseCalibrator(ABC):
    """Abstract interface for confidence calibration models."""

    @abstractmethod
    def calibrate(self, raw_score: float) -> float:
        """Transforms a raw fused score [0.0, 1.0] into a calibrated confidence [0.0, 1.0]."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns calibrator name, version, and parameter values."""
        pass


class IdentityCalibrator(BaseCalibrator):
    """Passthrough calibrator used when empirical calibration curve is uninitialized."""

    def __init__(self, version: str = "identity-v1.0"):
        self.version = version

    def calibrate(self, raw_score: float) -> float:
        return float(min(1.0, max(0.0, raw_score)))

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "calibrator_type": "Identity",
            "version": self.version,
            "parameters": {}
        }


class TemperatureCalibrator(BaseCalibrator):
    """
    Applies temperature scaling:
    p = sigmoid(logit(score) / T)
    """

    def __init__(self, temperature: float = 1.25, version: str = "temp-scale-v1.0"):
        self.temperature = max(1e-3, temperature)
        self.version = version

    def calibrate(self, raw_score: float) -> float:
        # Clip score away from 0.0 and 1.0 to prevent logit overflow
        s = min(0.9999, max(0.0001, raw_score))
        logit = np.log(s / (1.0 - s))
        scaled_logit = logit / self.temperature
        calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(min(1.0, max(0.0, calibrated)))

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "calibrator_type": "TemperatureScaling",
            "version": self.version,
            "parameters": {"temperature": self.temperature}
        }


class PlattCalibrator(BaseCalibrator):
    """
    Applies Platt Logistic Scaling:
    p = 1 / (1 + exp(A * score + B))
    """

    def __init__(self, a: float = -4.5, b: float = 2.25, version: str = "platt-mvp-v1.0"):
        """
        Default parameters represent empirical baseline sigmoid mapping score ~0.5 -> ~0.5.
        """
        self.a = float(a)
        self.b = float(b)
        self.version = version

    def calibrate(self, raw_score: float) -> float:
        s = min(1.0, max(0.0, raw_score))
        val = 1.0 / (1.0 + np.exp(self.a * s + self.b))
        return float(min(1.0, max(0.0, val)))

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "calibrator_type": "PlattScaling",
            "version": self.version,
            "parameters": {"a": self.a, "b": self.b}
        }
