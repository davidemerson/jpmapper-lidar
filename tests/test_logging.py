"""Tests for the logging module."""
import logging
import pytest
from unittest.mock import patch, MagicMock
import sys

from jpmapper.logging import get_logger, set_log_level


class TestLogging:
    """Test suite for the logging module."""
    
    @patch('jpmapper.logging.logging.getLogger')
    def test_get_logger(self, mock_get_logger):
        """Test that get_logger returns a logger with the correct name."""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Call get_logger
        logger = get_logger("test")
        
        # Verify that logging.getLogger was called with the correct name
        mock_get_logger.assert_called_once_with("jpmapper.test")
        
        # Verify that the returned logger is the mock logger
        assert logger == mock_logger
    
    @patch('jpmapper.logging.logging.getLogger')
    def test_get_logger_default_name(self, mock_get_logger):
        """Test that get_logger uses a default name if none is provided."""
        # Setup mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Call get_logger with no name
        logger = get_logger()
        
        # Verify that logging.getLogger was called with the default name
        mock_get_logger.assert_called_once_with("jpmapper")
        
        # Verify that the returned logger is the mock logger
        assert logger == mock_logger
    
    @patch('jpmapper.logging.logging')
    def test_set_log_level(self, mock_logging):
        """Test that set_log_level sets the correct log level."""
        # Call set_log_level
        set_log_level(logging.DEBUG)
        
        # Verify that basicConfig was called with the correct level
        mock_logging.basicConfig.assert_called_once()
        
        # Get the kwargs passed to basicConfig
        call_kwargs = mock_logging.basicConfig.call_args[1]
        
        # Verify that level=DEBUG was passed
        assert call_kwargs.get('level') == logging.DEBUG
    
    @patch('jpmapper.logging.logging')
    def test_set_log_level_with_stream(self, mock_logging):
        """Test that set_log_level sets the correct stream."""
        # Call set_log_level with a stream
        set_log_level(logging.INFO, stream=sys.stdout)
        
        # Verify that basicConfig was called with the correct stream
        mock_logging.basicConfig.assert_called_once()
        
        # Get the kwargs passed to basicConfig
        call_kwargs = mock_logging.basicConfig.call_args[1]
        
        # Verify that stream=sys.stdout was passed
        assert call_kwargs.get('stream') == sys.stdout
    
    @patch('jpmapper.logging.logging')
    def test_set_log_level_with_format(self, mock_logging):
        """Test that set_log_level sets the correct format."""
        # Call set_log_level with a custom format
        custom_format = "%(levelname)s: %(message)s"
        set_log_level(logging.WARNING, format=custom_format)
        
        # Verify that basicConfig was called with the correct format
        mock_logging.basicConfig.assert_called_once()
        
        # Get the kwargs passed to basicConfig
        call_kwargs = mock_logging.basicConfig.call_args[1]
        
        # Verify that format=custom_format was passed
        assert call_kwargs.get('format') == custom_format
