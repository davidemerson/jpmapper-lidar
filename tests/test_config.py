"""Tests for the configuration module."""
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from jpmapper.config import Config
from jpmapper.exceptions import ConfigurationError


class TestConfig:
    """Test suite for the Config class."""
    
    def test_default_config(self):
        """Test that the default config has expected values."""
        config = Config()
        
        # Check that default values are set
        assert hasattr(config, "debug")
        assert hasattr(config, "log_level")
        assert hasattr(config, "default_epsg")
        assert hasattr(config, "default_resolution")
    
    @patch('jpmapper.config.Path.exists')
    @patch('jpmapper.config.json.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_load_from_file(self, mock_open, mock_json_load, mock_exists):
        """Test loading configuration from a file."""
        # Setup mocks
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "debug": True,
            "log_level": "DEBUG",
            "default_epsg": 4326,
            "default_resolution": 0.5,
            "custom_key": "custom_value"
        }
        
        # Create a config instance that loads from the mocked file
        config = Config(config_path="config.json")
        
        # Check that values from the file were loaded
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.default_epsg == 4326
        assert config.default_resolution == 0.5
        assert config.custom_key == "custom_value"
    
    @patch('jpmapper.config.Path.exists')
    def test_file_not_found(self, mock_exists):
        """Test that loading a non-existent file raises ConfigurationError."""
        # Setup mock
        mock_exists.return_value = False
        
        # Check that trying to load a non-existent file raises ConfigurationError
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            Config(config_path="non_existent.json")
    
    @patch('jpmapper.config.Path.exists')
    @patch('builtins.open', side_effect=Exception("Error reading file"))
    def test_file_read_error(self, mock_open, mock_exists):
        """Test that an error reading the file raises ConfigurationError."""
        # Setup mock
        mock_exists.return_value = True
        
        # Check that an error reading the file raises ConfigurationError
        with pytest.raises(ConfigurationError, match="Error reading configuration file"):
            Config(config_path="config.json")
    
    @patch('jpmapper.config.Path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('jpmapper.config.json.load', side_effect=Exception("Invalid JSON"))
    def test_invalid_json(self, mock_json_load, mock_open, mock_exists):
        """Test that invalid JSON in the file raises ConfigurationError."""
        # Setup mock
        mock_exists.return_value = True
        
        # Check that invalid JSON raises ConfigurationError
        with pytest.raises(ConfigurationError, match="Invalid configuration file format"):
            Config(config_path="config.json")
    
    def test_dict_access(self):
        """Test dictionary-style access to config values."""
        config = Config()
        
        # Set a value
        config["test_key"] = "test_value"
        
        # Check dictionary-style access
        assert config["test_key"] == "test_value"
        
        # Check attribute-style access
        assert config.test_key == "test_value"
    
    def test_attribute_access(self):
        """Test attribute-style access to config values."""
        config = Config()
        
        # Set a value
        config.test_key = "test_value"
        
        # Check attribute-style access
        assert config.test_key == "test_value"
        
        # Check dictionary-style access
        assert config["test_key"] == "test_value"
    
    def test_contains(self):
        """Test the 'in' operator with config."""
        config = Config()
        
        # Set a value
        config.test_key = "test_value"
        
        # Check 'in' operator
        assert "test_key" in config
        assert "non_existent_key" not in config
    
    def test_get_with_default(self):
        """Test getting a value with a default."""
        config = Config()
        
        # Set a value
        config.existing_key = "existing_value"
        
        # Check get with default for existing key
        assert config.get("existing_key", "default") == "existing_value"
        
        # Check get with default for non-existent key
        assert config.get("non_existent_key", "default") == "default"
