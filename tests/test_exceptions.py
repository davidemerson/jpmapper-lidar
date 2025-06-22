"""Tests for jpmapper exception classes."""
import pytest

from jpmapper.exceptions import (
    JPMapperError,
    ConfigurationError,
    FileFormatError,
    GeoSpatialError,
    GeometryError,
    CRSError,
    NoDataError,
    AnalysisError,
    LOSError,
    RasterizationError,
    FilterError,
)


def test_exception_hierarchy():
    """Test the exception hierarchy is correct."""
    # Base exception
    assert issubclass(JPMapperError, Exception)
    
    # First level
    assert issubclass(ConfigurationError, JPMapperError)
    assert issubclass(FileFormatError, JPMapperError)
    assert issubclass(GeoSpatialError, JPMapperError)
    assert issubclass(AnalysisError, JPMapperError)
    assert issubclass(RasterizationError, JPMapperError)
    assert issubclass(FilterError, JPMapperError)
    
    # Second level
    assert issubclass(GeometryError, GeoSpatialError)
    assert issubclass(CRSError, GeoSpatialError)
    assert issubclass(NoDataError, GeoSpatialError)
    assert issubclass(LOSError, AnalysisError)


def test_exception_messages():
    """Test that exception messages are correctly stored."""
    error_msg = "Test error message"
    
    # Test base exception
    exc = JPMapperError(error_msg)
    assert str(exc) == error_msg
    
    # Test derived exceptions
    exceptions = [
        ConfigurationError,
        FileFormatError,
        GeoSpatialError,
        GeometryError,
        CRSError,
        NoDataError,
        AnalysisError,
        LOSError,
        RasterizationError,
        FilterError,
    ]
    
    for exc_class in exceptions:
        exc = exc_class(error_msg)
        assert str(exc) == error_msg
        assert isinstance(exc, JPMapperError)


def test_exception_chaining():
    """Test that exceptions can be chained."""
    original = ValueError("Original error")
    
    try:
        try:
            raise original
        except ValueError as e:
            raise GeometryError("Geometry error") from e
    except GeometryError as e:
        assert isinstance(e.__cause__, ValueError)
        assert str(e.__cause__) == "Original error"
