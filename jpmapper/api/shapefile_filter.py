"""
Enhanced filtering API with shapefile support for JPMapper
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List, Tuple, Union

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import fiona
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False

from shapely.geometry import Polygon, box
import pyproj
from pyproj import CRS, Transformer

from jpmapper.io.las import filter_las_by_bbox as _filter_las_by_bbox
from jpmapper.exceptions import FilterError, GeometryError, ConfigurationError


def filter_by_shapefile(
    las_files: Iterable[Path],
    shapefile_path: Path,
    *,
    dst_dir: Optional[Path] = None,
    buffer_meters: float = 0.0,
    validate_crs: bool = True,
) -> List[Path]:
    """
    Filter LAS/LAZ files using a shapefile boundary.
    
    Args:
        las_files: Iterable of Path objects pointing to LAS/LAZ files
        shapefile_path: Path to shapefile (.shp) containing boundary geometry
        dst_dir: Optional destination directory to copy filtered files
        buffer_meters: Buffer distance in meters to expand the shapefile boundary
        validate_crs: Whether to validate CRS compatibility between LAS and shapefile
        
    Returns:
        List of Path objects for files that intersect the shapefile boundary
        
    Raises:
        ConfigurationError: If required dependencies are not installed
        GeometryError: If shapefile cannot be read or processed
        FilterError: If filtering operation fails
        
    Example:
        >>> from pathlib import Path
        >>> from jpmapper.api import filter_by_shapefile
        >>> las_dir = Path("data/las")
        >>> boundary = Path("data/boundaries/study_area.shp")
        >>> filtered = filter_by_shapefile(
        ...     las_dir.glob("*.las"), 
        ...     boundary,
        ...     buffer_meters=50.0
        ... )
    """
    if not HAS_GEOPANDAS:
        raise ConfigurationError(
            "geopandas is required for shapefile support. "
            "Install with: conda install -c conda-forge geopandas"
        )
    
    if not HAS_FIONA:
        raise ConfigurationError(
            "fiona is required for shapefile support. "
            "Install with: conda install -c conda-forge fiona"
        )
    
    # Validate shapefile exists
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile does not exist: {shapefile_path}")
    
    try:
        # Read shapefile
        gdf = gpd.read_file(shapefile_path)
        
        if gdf.empty:
            raise GeometryError(f"Shapefile contains no geometries: {shapefile_path}")
        
        # Get the union of all geometries in the shapefile
        if len(gdf) > 1:
            # Combine multiple polygons into one
            boundary_geom = gdf.unary_union
        else:
            boundary_geom = gdf.geometry.iloc[0]
        
        # Apply buffer if specified
        if buffer_meters > 0:
            # Buffer in the shapefile's CRS (assuming it's in meters)
            boundary_geom = boundary_geom.buffer(buffer_meters)
        
        # Get shapefile CRS
        shapefile_crs = gdf.crs
        
        # Validate CRS compatibility if requested
        if validate_crs:
            _validate_crs_compatibility(las_files, shapefile_crs)
        
        # Filter LAS files using the shapefile geometry
        return _filter_las_by_geometry(
            las_files, 
            boundary_geom, 
            shapefile_crs,
            dst_dir=dst_dir
        )
        
    except Exception as e:
        if isinstance(e, (ConfigurationError, GeometryError, FilterError)):
            raise
        raise FilterError(f"Error processing shapefile {shapefile_path}: {e}") from e


def _validate_crs_compatibility(las_files: Iterable[Path], shapefile_crs: CRS) -> None:
    """Validate that LAS files and shapefile have compatible CRS."""
    import laspy
    
    # Sample a few LAS files to check CRS
    las_files_list = list(las_files)
    sample_files = las_files_list[:3]  # Check first 3 files
    
    for las_file in sample_files:
        if not las_file.exists():
            continue
            
        try:
            with laspy.open(str(las_file)) as las_reader:
                las_crs = las_reader.header.parse_crs()
                
            if las_crs is None:
                continue  # Skip files without CRS info
                
            # Check if CRS are the same or can be transformed
            if las_crs != shapefile_crs:
                # Try to create a transformer to validate compatibility
                try:
                    Transformer.from_crs(las_crs, shapefile_crs, always_xy=True)
                except Exception as e:
                    raise GeometryError(
                        f"Incompatible CRS: LAS file {las_file.name} uses {las_crs}, "
                        f"shapefile uses {shapefile_crs}. Error: {e}"
                    ) from e
                    
        except Exception as e:
            # Log warning but don't fail - some files might be problematic
            import logging
            logging.warning(f"Could not validate CRS for {las_file}: {e}")


def _filter_las_by_geometry(
    las_files: Iterable[Path],
    boundary_geom: Union[Polygon, any],  # Shapely geometry
    boundary_crs: CRS,
    *,
    dst_dir: Optional[Path] = None,
) -> List[Path]:
    """Filter LAS files using a shapely geometry."""
    import laspy
    from shapely.geometry import box
    
    selected: List[Path] = []
    errors: List[str] = []
    
    for las_file in las_files:
        if not las_file.exists():
            continue
            
        try:
            # Read LAS header to get bounds
            with laspy.open(str(las_file)) as las_reader:
                header = las_reader.header
                las_crs = header.parse_crs()
                
                # Get LAS file bounds
                las_bounds = box(
                    header.mins[0], header.mins[1],
                    header.maxs[0], header.maxs[1]
                )
                
                # Transform LAS bounds to shapefile CRS if needed
                if las_crs and las_crs != boundary_crs:
                    try:
                        transformer = Transformer.from_crs(
                            las_crs, boundary_crs, always_xy=True
                        )
                        
                        # Transform the corner points
                        min_x, min_y = transformer.transform(header.mins[0], header.mins[1])
                        max_x, max_y = transformer.transform(header.maxs[0], header.maxs[1])
                        
                        # Create new bounds in target CRS
                        las_bounds = box(min_x, min_y, max_x, max_y)
                        
                    except Exception as e:
                        errors.append(f"{las_file.name}: CRS transformation failed: {e}")
                        continue
                
                # Check for intersection with boundary
                if las_bounds.intersects(boundary_geom):
                    selected.append(las_file)
                    
        except Exception as e:
            errors.append(f"{las_file.name}: {e}")
            continue
    
    # Copy files if destination directory specified
    if dst_dir and selected:
        return _copy_files_to_destination(selected, dst_dir)
    
    return selected


def _copy_files_to_destination(src_files: List[Path], dst_dir: Path) -> List[Path]:
    """Copy selected files to destination directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    
    for src_file in src_files:
        dst_file = dst_dir / src_file.name
        try:
            dst_file.write_bytes(src_file.read_bytes())
            copied.append(dst_file)
        except Exception as e:
            raise FilterError(f"Failed to copy {src_file.name}: {e}") from e
    
    return copied


def create_boundary_from_las_files(
    las_files: Iterable[Path],
    output_shapefile: Path,
    *,
    buffer_meters: float = 0.0,
    epsg: Optional[int] = None,
) -> Path:
    """
    Create a boundary shapefile from the extents of LAS files.
    
    Args:
        las_files: LAS files to analyze for boundary creation
        output_shapefile: Path where the output shapefile will be written
        buffer_meters: Buffer distance to expand the boundary
        epsg: EPSG code for output CRS. If None, uses CRS from first LAS file.
        
    Returns:
        Path to the created shapefile
        
    Example:
        >>> create_boundary_from_las_files(
        ...     Path("data/las").glob("*.las"),
        ...     Path("data/boundary.shp"),
        ...     buffer_meters=100.0,
        ...     epsg=6539
        ... )
    """
    if not HAS_GEOPANDAS:
        raise ConfigurationError("geopandas is required for shapefile creation")
    
    import laspy
    from shapely.geometry import box
    
    # Collect all LAS file bounds
    all_bounds = []
    target_crs = None
    
    for las_file in las_files:
        if not las_file.exists():
            continue
            
        try:
            with laspy.open(str(las_file)) as las_reader:
                header = las_reader.header
                las_crs = header.parse_crs()
                
                # Set target CRS from first file if not specified
                if target_crs is None:
                    if epsg:
                        target_crs = CRS.from_epsg(epsg)
                    elif las_crs:
                        target_crs = las_crs
                    else:
                        target_crs = CRS.from_epsg(4326)  # Default to WGS84
                
                # Create bounds geometry
                bounds = box(
                    header.mins[0], header.mins[1],
                    header.maxs[0], header.maxs[1]
                )
                
                # Transform to target CRS if needed
                if las_crs and las_crs != target_crs:
                    transformer = Transformer.from_crs(las_crs, target_crs, always_xy=True)
                    min_x, min_y = transformer.transform(header.mins[0], header.mins[1])
                    max_x, max_y = transformer.transform(header.maxs[0], header.maxs[1])
                    bounds = box(min_x, min_y, max_x, max_y)
                
                all_bounds.append(bounds)
                
        except Exception as e:
            import logging
            logging.warning(f"Could not process {las_file}: {e}")
            continue
    
    if not all_bounds:
        raise FilterError("No valid LAS files found for boundary creation")
    
    # Create union of all bounds
    from shapely.ops import unary_union
    boundary = unary_union(all_bounds)
    
    # Apply buffer if specified
    if buffer_meters > 0:
        boundary = boundary.buffer(buffer_meters)
    
    # Create GeoDataFrame and save
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[boundary], crs=target_crs)
    output_shapefile.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_shapefile)
    
    return output_shapefile
