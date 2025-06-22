"""
JPMapper Exceptions
------------------

This module defines custom exceptions used throughout JPMapper.
Using specific exceptions makes error handling more precise and
helps users understand what went wrong.
"""

from __future__ import annotations


class JPMapperError(Exception):
    """Base exception for all JPMapper errors."""
    pass


class ConfigurationError(JPMapperError):
    """Raised when there's an issue with configuration settings."""
    pass


class FileFormatError(JPMapperError):
    """Raised when there's an issue with file formats or parsing."""
    pass


class GeoSpatialError(JPMapperError):
    """Base class for geospatial errors."""
    pass


class GeometryError(GeoSpatialError):
    """Raised when there's an issue with geometries or coordinates."""
    pass


class CRSError(GeoSpatialError):
    """Raised when there's an issue with coordinate reference systems."""
    pass


class NoDataError(GeoSpatialError):
    """Raised when required data is missing."""
    pass


class AnalysisError(JPMapperError):
    """Raised when an analysis operation fails."""
    pass


class LOSError(AnalysisError):
    """Raised when a line-of-sight analysis operation fails."""
    pass


class RasterizationError(JPMapperError):
    """Raised when a rasterization operation fails."""
    pass


class FilterError(JPMapperError):
    """Raised when a filtering operation fails."""
    pass
