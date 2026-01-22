from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import numpy as np
from gluonts.ev.aggregations import Mean
from gluonts.ev.metrics import BaseMetricDefinition, DirectMetric, squared_error, DerivedMetric
from gluonts.ev.aggregations import Sum

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Capture NumPy warnings
warnings.filterwarnings('error', category=RuntimeWarning)


@dataclass
class R2Score(BaseMetricDefinition):
    """Coefficient of determination (R^2 score) using DerivedMetric pattern."""

    forecast_type: str = "0.5"

    @staticmethod
    def compute_r2(
        sum_squared_error: np.ndarray,
        sum_squared_deviations_label: np.ndarray
    ) -> np.ndarray:
        """Post-process to calculate R² = 1 - (SS_res / SS_tot)"""
        logger.debug(f"compute_r2 input - SSE shape: {sum_squared_error.shape}, SSE: {sum_squared_error}")
        logger.debug(f"compute_r2 input - SS_deviations shape: {sum_squared_deviations_label.shape}, SS_deviations: {sum_squared_deviations_label}")
        
        # Check for NaN or Inf in inputs
        if np.any(np.isnan(sum_squared_error)):
            logger.warning("NaN detected in sum_squared_error input!")
        if np.any(np.isinf(sum_squared_error)):
            logger.warning("Inf detected in sum_squared_error input!")
        if np.any(np.isnan(sum_squared_deviations_label)):
            logger.warning("NaN detected in sum_squared_deviations_label input!")
        if np.any(np.isinf(sum_squared_deviations_label)):
            logger.warning("Inf detected in sum_squared_deviations_label input!")
        
        # Check for zero values (which could cause division issues)
        zero_count = np.sum(sum_squared_deviations_label == 0)
        if zero_count > 0:
            logger.warning(f"Zero values found in sum_squared_deviations_label: {zero_count} occurrences")
        
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                r2 = 1.0 - np.divide(sum_squared_error, sum_squared_deviations_label)
                logger.debug(f"R² before nan_to_num: {r2}")
                logger.debug(f"NaN count in R²: {np.sum(np.isnan(r2))}, Inf count: {np.sum(np.isinf(r2))}")
            
            result = np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0)
            logger.debug(f"R² after nan_to_num: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in compute_r2: {e}", exc_info=True)
            raise

    def __call__(self, axis: Optional[int] = None) -> DerivedMetric:
        return DerivedMetric(
            name=f"R2[{self.forecast_type}]",
            metrics={
                "sum_squared_error": SumSquaredError(forecast_type=self.forecast_type)(axis=axis),
                "sum_squared_deviations_label": SumSquaredDeviationsLabel()(axis=axis),
            },
            post_process=self.compute_r2,
        )
    
def sum_squared_deviations_label(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Sum of squared deviations from mean (SS_tot in terms of sum, not mean)."""
    try:
        label = data["label"]
        logger.debug(f"sum_squared_deviations_label - label shape: {label.shape}, label dtype: {label.dtype}")
        logger.debug(f"sum_squared_deviations_label - label values (first 10): {label.flat[:10]}")
        
        # Check for NaN or Inf in label
        if np.any(np.isnan(label)):
            logger.warning(f"NaN detected in label: {np.sum(np.isnan(label))} occurrences")
        if np.any(np.isinf(label)):
            logger.warning(f"Inf detected in label: {np.sum(np.isinf(label))} occurrences")
        
        mean_label = np.mean(label, axis=-1, keepdims=True)
        logger.debug(f"sum_squared_deviations_label - mean_label: {mean_label}")
        
        # Check for NaN/Inf in mean
        if np.any(np.isnan(mean_label)):
            logger.warning("NaN detected in mean_label!")
        if np.any(np.isinf(mean_label)):
            logger.warning("Inf detected in mean_label!")
        
        deviations = label - mean_label
        logger.debug(f"sum_squared_deviations_label - deviations (first 10): {deviations.flat[:10]}")
        
        squared_devs = np.square(deviations)
        logger.debug(f"sum_squared_deviations_label - squared_devs (first 10): {squared_devs.flat[:10]}")
        
        ss_deviations = np.sum(squared_devs, axis=-1, keepdims=True)
        logger.debug(f"sum_squared_deviations_label - sum result: {ss_deviations}")
        
        # Check for NaN/Inf in result
        if np.any(np.isnan(ss_deviations)):
            logger.warning(f"NaN detected in sum_squared_deviations_label result: {np.sum(np.isnan(ss_deviations))} occurrences")
        if np.any(np.isinf(ss_deviations)):
            logger.warning(f"Inf detected in sum_squared_deviations_label result: {np.sum(np.isinf(ss_deviations))} occurrences")
        
        return ss_deviations
    except Exception as e:
        logger.error(f"Error in sum_squared_deviations_label: {e}", exc_info=True)
        raise

    # Create the base metric definitions for the components
class SumSquaredDeviationsLabel(BaseMetricDefinition):
    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name="sum_squared_deviations_label",
            stat=sum_squared_deviations_label,
            aggregate=Sum(axis=axis),
        )


@dataclass
class SumSquaredError(BaseMetricDefinition):
    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"sum_squared_error[{self.forecast_type}]",
            stat=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )
