"""
Functions for analyzing terrain and line-of-sight.
"""
from pathlib import Path
import csv
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd

# Optional import for performance optimization
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from jpmapper.exceptions import AnalysisError, LOSError
from jpmapper.api import analyze_los
from jpmapper.io import raster as r

logger = logging.getLogger(__name__)


def _get_optimal_analysis_workers(workers: Optional[int] = None) -> int:
    """Get optimal number of workers for analysis tasks."""
    if workers is not None:
        return max(1, workers)
    
    # For analysis tasks, we can be more aggressive with CPU usage
    # since they're less memory-intensive than rasterization
    cpu_count = multiprocessing.cpu_count()
    
    # Use up to 90% of available CPUs for analysis
    optimal_workers = max(1, int(cpu_count * 0.9))
    logger.info(f"Auto-detected {optimal_workers} workers for analysis (CPU cores: {cpu_count})")
    return optimal_workers


def _analyze_single_row(args: Tuple[Dict[str, Any], Path, float, int]) -> Dict[str, Any]:
    """Analyze a single CSV row - used for parallel processing."""
    row, dsm_path, freq_ghz, max_mast_height_m = args
    
    try:
        # Extract coordinates
        point_a = (float(row.get("point_a_lat")), float(row.get("point_a_lon")))
        point_b = (float(row.get("point_b_lat")), float(row.get("point_b_lon")))
        
        # Analyze line of sight
        analysis = analyze_los(
            dsm_path, 
            point_a, 
            point_b, 
            freq_ghz=freq_ghz,
            max_mast_height_m=max_mast_height_m
        )
        
        # Create result entry
        result = {
            "id": row.get("id", f"link_{hash(str(row))}"),
            "point_a": point_a,
            "point_b": point_b,
            "clear": analysis["clear"],
            "mast_height_m": analysis["mast_height_m"],
            "distance_m": analysis["distance_m"],
            "surface_height_a_m": analysis.get("surface_height_a_m", 0),
            "surface_height_b_m": analysis.get("surface_height_b_m", 0),
            "clearance_min_m": analysis.get("clearance_min_m", 0),
            "freq_ghz": freq_ghz
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing row {row}: {e}")
        # Return a failed result instead of crashing
        return {
            "id": row.get("id", f"link_{hash(str(row))}"),
            "point_a": (0, 0),
            "point_b": (0, 0),
            "clear": False,
            "mast_height_m": -1,
            "distance_m": 0,
            "error": str(e)
        }


def analyze_csv_file(
    csv_path: Path,
    las_dir: Optional[Path] = None,
    cache: Optional[Path] = None,
    epsg: Optional[int] = None, 
    resolution: Optional[float] = None,
    workers: Optional[int] = None,
    max_mast_height_m: int = 5,
    output_format: str = "json",
    output_path: Optional[Path] = None,
    freq_ghz: float = 5.8
) -> List[Dict[str, Any]]:
    """
    Analyze points from a CSV file for line-of-sight visibility.
    
    Args:
        csv_path: Path to CSV file containing point coordinates
        las_dir: Optional directory with LAS files for on-the-fly DSM generation
        cache: Optional path to cache the DSM
        epsg: Optional EPSG code for the DSM
        resolution: Optional resolution in meters for the DSM
        workers: Optional number of workers for processing (auto-detected if None)
        max_mast_height_m: Maximum mast height to test in meters
        output_format: Output format (json, csv, geojson)
        output_path: Optional path to save results
        freq_ghz: Frequency in GHz for Fresnel zone calculation
        
    Returns:
        List of dictionaries with analysis results
        
    Raises:
        AnalysisError: If analysis fails
    """
    try:
        # Validate inputs
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Generate or use DSM
        dsm_path = None
        if cache and cache.exists():
            dsm_path = cache
            logger.info(f"Using cached DSM: {dsm_path}")
        elif las_dir:
            if cache is None:
                cache = las_dir.parent / "dsm_cache.tif"
            dsm_path = r.cached_mosaic(
                las_dir, 
                cache, 
                epsg=epsg or 6539, 
                resolution=resolution or 0.1,
                workers=workers  # Pass workers to rasterization too
            )
            logger.info(f"Generated DSM from LAS files: {dsm_path}")
        else:
            raise ValueError("Either cache or las_dir must be provided")
            
        # Read CSV file into memory
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        logger.info(f"Processing {len(rows)} point pairs from CSV")
        
        # Get optimal number of workers for analysis
        analysis_workers = _get_optimal_analysis_workers(workers)
        
        # Process rows in parallel if we have multiple workers and multiple rows
        if analysis_workers > 1 and len(rows) > 1:
            logger.info(f"Using {analysis_workers} workers for parallel analysis")
            
            # Prepare arguments for parallel processing
            args_list = [(row, dsm_path, freq_ghz, max_mast_height_m) for row in rows]
            
            results = []
            with ProcessPoolExecutor(max_workers=analysis_workers) as executor:
                # Submit all tasks
                future_to_row = {
                    executor.submit(_analyze_single_row, args): i 
                    for i, args in enumerate(args_list)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_row):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        row_idx = future_to_row[future]
                        logger.error(f"Error processing row {row_idx}: {e}")
                        # Add a failed result
                        results.append({
                            "id": f"failed_row_{row_idx}",
                            "error": str(e),
                            "clear": False,
                            "mast_height_m": -1
                        })
        else:
            # Sequential processing for single worker or single row
            results = []
            for row in rows:
                try:
                    args = (row, dsm_path, freq_ghz, max_mast_height_m)
                    result = _analyze_single_row(args)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing row {row}: {e}")
                    results.append({
                        "id": row.get("id", f"failed_{len(results)}"),
                        "error": str(e),
                        "clear": False,
                        "mast_height_m": -1
                    })
        
        # Save results if output path specified
        if output_path:
            if output_format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            elif output_format.lower() == "csv":
                if results:
                    df = pd.DataFrame(results)
                    df.to_csv(output_path, index=False)
            
        return results
        
    except Exception as e:
        raise AnalysisError(f"Failed to analyze CSV file: {e}") from e
