from pathlib import Path
import yaml
from typing import Dict, Any

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data Sources
MLB_API_BASE_URL = "https://statsapi.mlb.com/api/v1"

# Transformation/Load Details
START_DATE = "2019-03-20"  # Start of 2019 season
END_DATE = "2024-09-29"    # End of 2024 season

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML. If None, uses default.
        
    Returns:
        Dictionary of configuration parameters.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}
