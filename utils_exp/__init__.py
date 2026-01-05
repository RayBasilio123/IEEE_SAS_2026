import warnings

from utils_exp.cli import Args, get_parser
from utils_exp.experiment_facade import MLExperimentFacade
import logging
import os

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.DEBUG))


warnings.filterwarnings("ignore",
                        message="'M' is deprecated and will be removed in a future version, please use 'ME' instead.")

warnings.filterwarnings("ignore",
                        message="The mean prediction is not stored in the forecast data; the median is being returned instead. This behaviour may change in the future.")

__all__ = ["MLExperimentFacade", "get_parser", "Args"]
