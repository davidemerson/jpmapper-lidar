"""
Functions for analyzing terrain and line-of-sight.
"""
from pathlib import Path
import csv
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd

from jpmapper.exceptions import AnalysisError, LOSError
from jpmapper.api import analyze_los
from jpmapper.io import raster as r

logger = logging.getLogger(__name__)


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
        workers: Optional number of workers for processing
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
                workers=workers
            )
            logger.info(f"Generated DSM from LAS files: {dsm_path}")
        else:
            raise ValueError("Either cache or las_dir must be provided")
            
        # Read CSV file
        results = []
        
        # Process each row in the CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                        "id": row.get("id", f"link_{len(results) + 1}"),
                        "point_a": point_a,
                        "point_b": point_b,
                        "clear": analysis["clear"],
                        "mast_height_m": analysis["mast_height_m"],
                        "distance_m": analysis.get("distance_m", 0),
                        "ground_a": analysis["ground_a_m"],
                        "ground_b": analysis["ground_b_m"]
                    }
                    
                    results.append(result)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error processing row: {e}")
                    continue
                except LOSError as e:
                    logger.warning(f"LOS analysis error: {e}")
                    continue
        
        # Save output if requested
        if output_path:
            if output_format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
            elif output_format.lower() == "csv":                # Create a flattened DataFrame from results
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
            elif output_format.lower() == "geojson":                # Create GeoJSON features
                features = []
                for link in results:
                    lat_a, lon_a = link["point_a"]
                    lat_b, lon_b = link["point_b"]
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "id": link["id"],
                            "clear": link["clear"],
                            "mast_height_m": link["mast_height_m"],
                            "distance_m": link.get("distance_m", 0),
                            "ground_a_m": link["ground_a"],
                            "ground_b_m": link["ground_b"]
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [lon_a, lat_a],  # GeoJSON uses [lon, lat] order
                                [lon_b, lat_b]
                            ]
                        }
                    })
                
                geojson = {
                    "type": "FeatureCollection",
                    "features": features
                }
                
                with open(output_path, 'w') as f:
                    json.dump(geojson, f, indent=2)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        return results
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise AnalysisError(f"Error analyzing CSV file: {e}") from e
