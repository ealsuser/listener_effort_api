from dataclasses import dataclass

@dataclass
class LoggingConfig:
    """
    Configuration for logging.
    """
    level: str = "INFO"
    format: str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

@dataclass
class Models:
    models_path: str = "/home/ubuntu/git/LEPM_API/lepm_api/models/"