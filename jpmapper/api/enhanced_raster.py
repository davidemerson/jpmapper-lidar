"""
Enhanced rasterization API that leverages metadata for improved reliability.

This module extends the existing rasterization capabilities with metadata-aware
features to ensure more accurate and reliable GeoTIFF output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging

from jpmapper.io.raster import rasterize_tile as _base_rasterize_tile
from jpmapper.io.metadata_raster import MetadataAwareRasterizer
from jpmapper.exceptions import RasterizationError, CRSError

log = logging.getLogger(__name__)


def rasterize_tile_with_metadata(
    src_las: Path,
    dst_tif: Path,
    *,
    epsg: Optional[int] = None,
    resolution: Optional[float] = None,
    use_metadata: bool = True,
    metadata_dir: Optional[Path] = None,
    auto_adjust_resolution: bool = True,
    quality_threshold: Optional[float] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Rasterize a LAS file with metadata-enhanced configuration.
    
    This function improves upon the standard rasterization by:
    1. Auto-detecting CRS from associated .prj files or shapefiles
    2. Using tile boundary information for better spatial alignment
    3. Adjusting resolution based on dataset accuracy metrics
    4. Providing detailed metadata about the rasterization process
    
    Args:
        src_las: Path to the source LAS/LAZ file
        dst_tif: Path where the output GeoTIFF will be written
        epsg: EPSG code for output CRS. If None, auto-detects from metadata.
        resolution: Cell size in meters. If None, determines optimal based on accuracy.
        use_metadata: Whether to use associated metadata files for enhancement
        metadata_dir: Directory containing metadata files (if different from LAS location)
        auto_adjust_resolution: Whether to adjust resolution based on accuracy data
        quality_threshold: Minimum quality threshold for processing (in same units as accuracy)
        
    Returns:
        Tuple of (output_path, metadata_info_dict)
        
    Raises:
        FileNotFoundError: If src_las does not exist
        CRSError: If CRS cannot be determined and epsg is None
        RasterizationError: If rasterization fails
        
    Example:
        >>> from pathlib import Path
        >>> from jpmapper.api import rasterize_tile_with_metadata
        >>> result_path, metadata = rasterize_tile_with_metadata(
        ...     Path("data/las/tile.las"),
        ...     Path("output/tile.tif"),
        ...     use_metadata=True,
        ...     auto_adjust_resolution=True
        ... )
        >>> print(f"CRS source: {metadata['crs_source']}")
        >>> print(f"Resolution used: {metadata['resolution']} m")
        >>> print(f"Accuracy info: {metadata.get('accuracy_summary', 'None')}")
    """
    # Validate inputs
    if not src_las.exists():
        raise FileNotFoundError(f"Source LAS file does not exist: {src_las}")
    
    # Initialize metadata info dictionary
    metadata_info = {
        'source_file': str(src_las),
        'output_file': str(dst_tif),
        'use_metadata': use_metadata,
        'metadata_enhanced': False
    }
    
    if use_metadata:
        try:
            # Create metadata-aware rasterizer
            rasterizer = MetadataAwareRasterizer(metadata_dir)
            
            # Enhanced rasterization with metadata
            result_path, enhanced_metadata = rasterizer.enhanced_rasterize(
                src_las,
                dst_tif,
                epsg=epsg,
                resolution=resolution or 0.1,  # Default resolution if not provided
                use_metadata=True
            )
            
            # Merge metadata information
            metadata_info.update(enhanced_metadata)
            metadata_info['metadata_enhanced'] = True
            
            # Auto-adjust resolution if requested and accuracy data is available
            if auto_adjust_resolution and resolution is None and 'accuracy_info' in enhanced_metadata:
                optimal_resolution = _calculate_optimal_resolution(enhanced_metadata['accuracy_info'])
                if optimal_resolution and optimal_resolution != enhanced_metadata.get('resolution', 0.1):
                    log.info(f"Re-rasterizing with optimal resolution: {optimal_resolution}m")
                    
                    # Re-rasterize with optimal resolution
                    result_path, final_metadata = rasterizer.enhanced_rasterize(
                        src_las,
                        dst_tif,
                        epsg=enhanced_metadata.get('used_epsg'),
                        resolution=optimal_resolution,
                        use_metadata=True
                    )
                    
                    metadata_info.update(final_metadata)
                    metadata_info['resolution_optimized'] = True
                    metadata_info['original_resolution'] = 0.1
                    metadata_info['optimal_resolution'] = optimal_resolution
            
            # Check quality threshold if specified
            if quality_threshold and 'accuracy_info' in enhanced_metadata:
                quality_check = _check_quality_threshold(enhanced_metadata['accuracy_info'], quality_threshold)
                metadata_info['quality_check'] = quality_check
                if not quality_check['meets_threshold']:
                    log.warning(f"Quality below threshold: {quality_check['message']}")
            
            return result_path, metadata_info
            
        except Exception as e:
            log.warning(f"Metadata-enhanced rasterization failed, falling back to standard: {e}")
            metadata_info['metadata_error'] = str(e)
            metadata_info['fallback_used'] = True
    
    # Fallback to standard rasterization
    try:
        result_path = _base_rasterize_tile(
            src_las,
            dst_tif,
            epsg=epsg,
            resolution=resolution or 0.1
        )
        
        metadata_info['output_path'] = result_path
        metadata_info['used_epsg'] = epsg or 'auto-detected'
        metadata_info['resolution'] = resolution or 0.1
        metadata_info['method'] = 'standard_rasterization'
        
        return result_path, metadata_info
        
    except Exception as e:
        raise RasterizationError(f"Failed to rasterize {src_las}: {e}") from e


def _calculate_optimal_resolution(accuracy_info: Dict[str, Any]) -> Optional[float]:
    """
    Calculate optimal resolution based on accuracy information.
    
    Args:
        accuracy_info: Dictionary containing accuracy metrics
        
    Returns:
        Optimal resolution in meters, or None if cannot be determined
    """
    try:
        # Look for vertical accuracy (NVA) first
        if 'NVA_vertical' in accuracy_info:
            nva = accuracy_info['NVA_vertical']
            value = nva['value']
            unit = nva.get('unit', 'unknown')
            
            # Convert to meters if needed
            if unit in ['ft_US', 'ftUS', 'us_survey_foot']:
                value_m = value * 0.3048006096012192
            elif unit in ['ft', 'foot']:
                value_m = value * 0.3048
            else:
                value_m = value  # Assume meters
            
            # Use 2x the vertical accuracy as optimal resolution
            # This ensures we're not over-sampling beyond the data's inherent precision
            optimal_res = max(0.05, value_m * 2)  # Minimum 5cm resolution
            
            log.info(f"Calculated optimal resolution: {optimal_res:.3f}m based on NVA: {value} {unit}")
            return round(optimal_res, 3)
        
        # Fallback to RMSE if available
        if 'RMSE_relative' in accuracy_info:
            rmse = accuracy_info['RMSE_relative']
            value = rmse['value']
            
            # Convert to meters (assuming ft_US)
            value_m = value * 0.3048006096012192
            optimal_res = max(0.05, value_m * 1.5)
            
            log.info(f"Calculated optimal resolution: {optimal_res:.3f}m based on RMSE: {value} ft_US")
            return round(optimal_res, 3)
            
    except Exception as e:
        log.warning(f"Could not calculate optimal resolution: {e}")
    
    return None


def _check_quality_threshold(accuracy_info: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """
    Check if the dataset meets a quality threshold.
    
    Args:
        accuracy_info: Dictionary containing accuracy metrics
        threshold: Quality threshold value
        
    Returns:
        Dictionary with quality check results
    """
    result = {
        'meets_threshold': False,
        'threshold': threshold,
        'best_accuracy': None,
        'message': 'No accuracy data available'
    }
    
    try:
        # Check various accuracy metrics
        accuracies = []
        
        for key, acc in accuracy_info.items():
            if isinstance(acc, dict) and 'value' in acc:
                value = acc['value']
                unit = acc.get('unit', 'unknown')
                
                # Convert to meters if needed
                if unit in ['ft_US', 'ftUS', 'us_survey_foot']:
                    value_m = value * 0.3048006096012192
                elif unit in ['ft', 'foot']:
                    value_m = value * 0.3048
                else:
                    value_m = value  # Assume meters
                
                accuracies.append((key, value_m, value, unit))
        
        if accuracies:
            # Find the best (smallest) accuracy value
            best = min(accuracies, key=lambda x: x[1])
            result['best_accuracy'] = {
                'metric': best[0],
                'value_m': best[1],
                'original_value': best[2],
                'unit': best[3]
            }
            
            result['meets_threshold'] = best[1] <= threshold
            result['message'] = f"Best accuracy: {best[1]:.3f}m ({best[0]}), threshold: {threshold}m"
        
    except Exception as e:
        result['message'] = f"Error checking quality: {e}"
    
    return result


def batch_rasterize_with_metadata(
    las_files: List[Path],
    output_dir: Path,
    *,
    epsg: Optional[int] = None,
    resolution: Optional[float] = None,
    use_metadata: bool = True,
    metadata_dir: Optional[Path] = None,
    auto_adjust_resolution: bool = True,
    quality_threshold: Optional[float] = None,
    max_workers: Optional[int] = None
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Batch rasterize multiple LAS files with metadata enhancement.
    
    Args:
        las_files: List of LAS file paths to process
        output_dir: Directory where output GeoTIFFs will be written
        epsg: EPSG code for output CRS
        resolution: Cell size in meters
        use_metadata: Whether to use metadata enhancement
        metadata_dir: Directory containing metadata files
        auto_adjust_resolution: Whether to optimize resolution per file
        quality_threshold: Quality threshold for processing
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of (output_path, metadata_info) tuples
        
    Example:
        >>> from pathlib import Path
        >>> from jpmapper.api import batch_rasterize_with_metadata
        >>> las_files = list(Path("data/las").glob("*.las"))
        >>> results = batch_rasterize_with_metadata(
        ...     las_files,
        ...     Path("output/dsm"),
        ...     use_metadata=True,
        ...     auto_adjust_resolution=True
        ... )
        >>> for output_path, metadata in results:
        ...     print(f"Processed: {output_path.name} with {metadata['method']}")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    failed_files = []
    
    for las_file in las_files:
        try:
            output_file = output_dir / f"{las_file.stem}.tif"
            
            result_path, metadata = rasterize_tile_with_metadata(
                las_file,
                output_file,
                epsg=epsg,
                resolution=resolution,
                use_metadata=use_metadata,
                metadata_dir=metadata_dir,
                auto_adjust_resolution=auto_adjust_resolution,
                quality_threshold=quality_threshold
            )
            
            results.append((result_path, metadata))
            log.info(f"Successfully processed: {las_file.name}")
            
        except Exception as e:
            log.error(f"Failed to process {las_file.name}: {e}")
            failed_files.append((las_file, str(e)))
    
    if failed_files:
        log.warning(f"Failed to process {len(failed_files)} files")
        for failed_file, error in failed_files:
            log.warning(f"  {failed_file.name}: {error}")
    
    log.info(f"Batch processing complete: {len(results)} successful, {len(failed_files)} failed")
    return results


def generate_processing_report(
    results: List[Tuple[Path, Dict[str, Any]]],
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive report from batch processing results.
    
    Args:
        results: Results from batch_rasterize_with_metadata
        output_file: Optional file to write the report to
        
    Returns:
        Report dictionary
    """
    report = {
        'summary': {
            'total_files': len(results),
            'metadata_enhanced': 0,
            'resolution_optimized': 0,
            'quality_issues': 0,
            'crs_sources': {},
            'resolution_distribution': {}
        },
        'files': [],
        'recommendations': []
    }
    
    for output_path, metadata in results:
        file_info = {
            'file': output_path.name,
            'source': metadata.get('source_file', ''),
            'metadata_enhanced': metadata.get('metadata_enhanced', False),
            'crs_source': metadata.get('crs_source', 'unknown'),
            'resolution': metadata.get('resolution', 0),
            'epsg': metadata.get('used_epsg', 'unknown')
        }
        
        # Update summary statistics
        if metadata.get('metadata_enhanced'):
            report['summary']['metadata_enhanced'] += 1
        
        if metadata.get('resolution_optimized'):
            report['summary']['resolution_optimized'] += 1
        
        crs_source = metadata.get('crs_source', 'unknown')
        report['summary']['crs_sources'][crs_source] = report['summary']['crs_sources'].get(crs_source, 0) + 1
        
        resolution = metadata.get('resolution', 0)
        res_key = f"{resolution:.3f}m"
        report['summary']['resolution_distribution'][res_key] = report['summary']['resolution_distribution'].get(res_key, 0) + 1
        
        # Check for quality issues
        quality_check = metadata.get('quality_check')
        if quality_check and not quality_check.get('meets_threshold', True):
            report['summary']['quality_issues'] += 1
            file_info['quality_issue'] = quality_check.get('message', 'Quality below threshold')
        
        report['files'].append(file_info)
    
    # Generate recommendations
    if report['summary']['metadata_enhanced'] < report['summary']['total_files']:
        report['recommendations'].append(
            f"Consider ensuring metadata files (.prj, .shp, .xml) are available for all LAS files. "
            f"Only {report['summary']['metadata_enhanced']}/{report['summary']['total_files']} files used metadata enhancement."
        )
    
    if report['summary']['resolution_optimized'] > 0:
        report['recommendations'].append(
            f"{report['summary']['resolution_optimized']} files had their resolution optimized based on accuracy data. "
            f"This can improve output quality while avoiding over-sampling."
        )
    
    if report['summary']['quality_issues'] > 0:
        report['recommendations'].append(
            f"{report['summary']['quality_issues']} files had quality issues. "
            f"Review accuracy metrics and consider adjusting quality thresholds."
        )
    
    # Write report to file if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        log.info(f"Processing report written to: {output_file}")
    
    return report
