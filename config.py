from dataclasses import dataclass
from pathlib import Path


@dataclass
class Models:
    models_path: str = str(Path(__file__).parent / "models")

@dataclass
class LoggingConfig:
    """
    Configuration for logging.
    """
    level: str = "INFO"
    format: str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
