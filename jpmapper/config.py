"""Load configuration from config files, env vars, and defaults."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from jpmapper.exceptions import ConfigurationError

_CONFIG_ENV = "JPMAPPER_CONFIG_FILE"
_DEFAULT_PATH = Path.home() / ".jpmapper.json"


class Config:
    """
    Configuration manager for JPMapper.
    
    This class manages configuration settings for the application, with support for
    loading from JSON files, environment variables, and default values.
    
    Attributes can be accessed using both attribute-style (config.debug) and
    dictionary-style (config["debug"]) notation.
    
    Args:
        config_path: Optional path to a configuration file to load
        
    Raises:
        ConfigurationError: If there's an error loading or parsing the configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration values
        self._config: Dict[str, Any] = {
            "debug": False,
            "log_level": "INFO",
            "default_epsg": 4326,  # WGS84
            "default_resolution": 1.0,  # 1 meter
            "bbox": (-74.47, 40.48, -73.35, 41.03),  # NYC default
            "epsg": 6539  # NY‑Long‐Island ftUS
        }
          # Load from file if specified
        if config_path:
            self._load_from_file(config_path)
            
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from a JSON file."""
        path = Path(config_path)
        
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        
        try:
            # Use built-in open to support proper mocking in tests
            with open(path, "r") as f:
                try:
                    file_config = json.load(f)
                    self._config.update(file_config)
                except Exception as e:
                    raise ConfigurationError(f"Invalid configuration file format: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")
    
    def __getattr__(self, name: str) -> Any:
        """Support attribute-style access: config.debug"""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Support attribute-style setting: config.debug = True"""
        if name == "_config":
            super().__setattr__(name, value)
        else:
            self._config[name] = value
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access: config["debug"]"""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style setting: config["debug"] = True"""
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: "debug" in config"""
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        return self._config.get(key, default)
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            path: Path to save the configuration file to. If None, uses the default path.
            
        Raises:
            ConfigurationError: If there's an error saving the configuration
        """
        if path is None:
            path = _DEFAULT_PATH
        
        path = Path(path)
        
        try:
            with path.open("w") as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration file: {e}")


# Global configuration loading function
def load(config_path: Optional[str] = None) -> Config:
    """
    Load a configuration from the environment or default location.
    
    Order of precedence:
    1. Explicit config_path parameter
    2. JPMAPPER_CONFIG_FILE environment variable
    3. ~/.jpmapper.json
    4. Default values
    
    Args:
        config_path: Optional explicit path to a configuration file
        
    Returns:
        Loaded configuration object
        
    Raises:
        ConfigurationError: If there's an error loading the configuration
    """
    # Use explicit path if provided
    if config_path:
        return Config(config_path)
    
    # Check environment variable
    env_path = os.environ.get(_CONFIG_ENV)
    if env_path:
        return Config(env_path)
    
    # Check default path
    if _DEFAULT_PATH.exists():
        return Config(str(_DEFAULT_PATH))
    
    # Return default configuration
    return Config()