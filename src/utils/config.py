from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_config(folder: str="config", file: str="config.yaml") -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        folder(str): The folder where the configuration file is located. Default is "config".
        file(str): The name of configuration file. Default is "config.yaml".

    Returns:
        dict: A dictionary containing the configuration parameters.
        """

    config_path = BASE_DIR / folder / file

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for key, path in config["paths"].items():
        # convert relative paths to absolute paths
        if not Path(path).is_absolute():
            config["paths"][key] = str(BASE_DIR / path)
    config["project_root"] = str(BASE_DIR)
    
    return config
