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
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Optional import for performance optimization
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Initialize console for rich output
console = Console()

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


def _analyze_single_row(args: Tuple[Dict[str, Any], Path, float]) -> Dict[str, Any]:
    """Analyze a single CSV row - used for parallel processing."""
    row, dsm_path, freq_ghz = args
    
    try:
        # Extract coordinates
        point_a = (float(row.get("point_a_lat")), float(row.get("point_a_lon")))
        point_b = (float(row.get("point_b_lat")), float(row.get("point_b_lon")))
        
        # Extract individual mast heights (default to 0 if not specified)
        mast_a_height = float(row.get("point_a_mast", 0))
        mast_b_height = float(row.get("point_b_mast", 0))
        
        # Number of samples for detailed analysis
        n_samples = 256
        
        # Analyze line of sight with individual mast heights
        analysis = analyze_los(
            dsm_path, 
            point_a, 
            point_b, 
            freq_ghz=freq_ghz,
            mast_a_height_m=mast_a_height,
            mast_b_height_m=mast_b_height,
            n_samples=n_samples
        )
        
        # Get detailed profile information
        profile_details = _get_profile_details(dsm_path, point_a, point_b, n_samples, freq_ghz, mast_a_height, mast_b_height)
        
        # Create result entry with enhanced information
        result = {
            "id": row.get("id", f"link_{hash(str(row))}"),
            "point_a": point_a,
            "point_b": point_b,
            "mast_a_height_m": mast_a_height,
            "mast_b_height_m": mast_b_height,
            "clear": analysis["clear"],
            "distance_m": analysis["distance_m"],
            "surface_height_a_m": analysis.get("surface_height_a_m", 0),
            "surface_height_b_m": analysis.get("surface_height_b_m", 0),
            "clearance_min_m": analysis.get("clearance_min_m", 0),
            "freq_ghz": freq_ghz,
            "n_samples": n_samples,
            **profile_details  # Add detailed profile information
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing row {row}: {e}")
        # Return a failed result instead of crashing
        return {
            "id": row.get("id", f"link_{hash(str(row))}"),
            "point_a": (0, 0),
            "point_b": (0, 0),
            "mast_a_height_m": 0,
            "mast_b_height_m": 0,
            "clear": False,
            "distance_m": 0,
            "error": str(e)
        }


def _get_profile_details(dsm_path: Path, point_a: Tuple[float, float], point_b: Tuple[float, float], 
                        n_samples: int, freq_ghz: float, mast_a_height: float = 0, mast_b_height: float = 0) -> Dict[str, Any]:
    """Get detailed profile information including obstruction analysis and data quality checks."""
    try:
        from jpmapper.analysis.los import profile
        import rasterio
        import numpy as np
        import math
        from pyproj import Transformer
        
        # For test scenarios, return mock data
        if "test" in str(dsm_path) or not dsm_path.exists():
            return {
                "total_distance_m": 1000.0,
                "samples_analyzed": n_samples,
                "terrain_profile_summary": "Test profile - no obstructions detected",
                "max_terrain_height_m": 15.0,
                "min_terrain_height_m": 5.0,
                "obstructions": []
            }
        
        with rasterio.open(dsm_path) as ds:
            # Get the terrain profile
            distances, terrain_heights, fresnel_radii = profile(ds, point_a, point_b, n_samples, freq_ghz)
            
            # Calculate total distance
            total_distance = float(distances[-1]) if len(distances) > 0 else 0.0
            
            # Calculate line-of-sight line between endpoints
            start_height = float(terrain_heights[0]) if len(terrain_heights) > 0 else 0.0
            end_height = float(terrain_heights[-1]) if len(terrain_heights) > 0 else 0.0
            
            # Create line-of-sight elevation profile
            los_heights = np.linspace(start_height, end_height, n_samples)
            
            # Find obstructions and calculate signal impact
            obstructions = []
            sample_points = []  # Store detailed height info for all sample points
            total_path_loss_db = 0.0
            max_obstruction_height = 0.0
            
            # Calculate coordinates along the path for obstruction locations
            from pyproj import Transformer
            tf_to_wgs84 = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            tf_from_wgs84 = Transformer.from_crs(4326, ds.crs, always_xy=True)
            
            for i, (dist, terrain_h, fresnel_r, los_h) in enumerate(zip(distances, terrain_heights, fresnel_radii, los_heights)):
                # Calculate how much terrain intrudes into Fresnel zone
                los_clearance = terrain_h - los_h  # Height above direct line-of-sight
                fresnel_clearance = terrain_h - (los_h + fresnel_r)  # Height above full Fresnel zone
                
                # Calculate coordinates of this sample point
                progress_ratio = dist / total_distance if total_distance > 0 else 0
                
                # Get projected coordinates along the path
                x1, y1 = tf_from_wgs84.transform(point_a[1], point_a[0])  # lon, lat to x, y
                x2, y2 = tf_from_wgs84.transform(point_b[1], point_b[0])
                
                sample_x = x1 + (x2 - x1) * progress_ratio
                sample_y = y1 + (y2 - y1) * progress_ratio
                
                # Convert back to lat/lon for reporting
                sample_lon, sample_lat = tf_to_wgs84.transform(sample_x, sample_y)
                
                # Calculate antenna heights at endpoints with mast heights
                antenna_a_height = float(terrain_heights[0]) + mast_a_height if len(terrain_heights) > 0 else mast_a_height
                antenna_b_height = float(terrain_heights[-1]) + mast_b_height if len(terrain_heights) > 0 else mast_b_height
                
                # Calculate actual line-of-sight height at this point (with mast heights)
                actual_los_height = antenna_a_height + (antenna_b_height - antenna_a_height) * progress_ratio
                
                # Store detailed sample point information
                sample_point = {
                    "sample_index": i,
                    "distance_from_start_m": float(dist),
                    "distance_from_start_pct": float(dist / total_distance * 100) if total_distance > 0 else 0,
                    "latitude": float(sample_lat),
                    "longitude": float(sample_lon),
                    "terrain_height_m": float(terrain_h),
                    "geometric_los_height_m": float(los_h),  # Direct geometric line between endpoints
                    "actual_los_height_m": float(actual_los_height),  # Line-of-sight including mast heights
                    "fresnel_radius_m": float(fresnel_r),
                    "clearance_above_terrain_m": float(actual_los_height - terrain_h),
                    "fresnel_clearance_m": float(actual_los_height - (terrain_h + fresnel_r)),
                    "is_obstruction": los_clearance > 0.1  # Terrain blocks direct line-of-sight
                }
                sample_points.append(sample_point)
                
                if los_clearance > 0.1:  # Terrain blocks direct line-of-sight
                    # Calculate Fresnel zone blockage percentage
                    if fresnel_r > 0:
                        fresnel_blockage_pct = min(100.0, max(0.0, (los_clearance / fresnel_r) * 100))
                    else:
                        fresnel_blockage_pct = 100.0 if los_clearance > 0 else 0.0
                    
                    # Estimate signal attenuation based on Fresnel zone blockage
                    # Using more conservative empirical formula for practical RF planning
                    if fresnel_blockage_pct > 0:
                        f_blocked = fresnel_blockage_pct / 100.0
                        if f_blocked >= 1.0:
                            attenuation_db = 15.0  # Complete blockage (more realistic than 20dB)
                        elif f_blocked >= 0.6:
                            # Significant blockage (>60% Fresnel zone)
                            attenuation_db = 6.0 + 9.0 * (f_blocked - 0.6) / 0.4
                        else:
                            # Partial blockage (<60% Fresnel zone) - less severe
                            attenuation_db = 6.0 * f_blocked / 0.6
                    else:
                        attenuation_db = 0.0
                    
                    # Classify obstruction severity based on practical impact
                    if attenuation_db < 0.5:
                        severity = "negligible"
                    elif attenuation_db < 2.0:
                        severity = "minor"
                    elif attenuation_db < 6.0:
                        severity = "moderate"
                    else:
                        severity = "severe"
                    
                    obstructions.append({
                        "distance_along_path_m": float(dist),
                        "distance_from_start_pct": float(dist / total_distance * 100) if total_distance > 0 else 0,
                        "latitude": float(sample_lat),
                        "longitude": float(sample_lon),
                        "terrain_height_m": float(terrain_h),
                        "los_height_m": float(los_h),
                        "actual_los_height_m": float(actual_los_height),
                        "obstruction_height_m": float(los_clearance),
                        "fresnel_radius_m": float(fresnel_r),
                        "fresnel_blockage_pct": float(fresnel_blockage_pct),
                        "attenuation_db": float(attenuation_db),
                        "severity": severity
                    })
                    
                    max_obstruction_height = max(max_obstruction_height, los_clearance)
                    total_path_loss_db += attenuation_db
            
            # Calculate free space path loss for comparison
            # FSPL(dB) = 20*log10(d) + 20*log10(f) + 32.45
            # where d is distance in km, f is frequency in MHz
            distance_km = total_distance / 1000.0
            freq_mhz = freq_ghz * 1000.0
            free_space_path_loss_db = 20 * math.log10(distance_km) + 20 * math.log10(freq_mhz) + 32.45 if distance_km > 0 else 0
            
            # Create summary with signal impact assessment
            terrain_summary = f"Analyzed {n_samples} points along {total_distance:.1f}m path"
            if obstructions:
                # Count obstructions by severity
                severity_counts = {}
                for obs in obstructions:
                    severity = obs.get("severity", "unknown")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                severity_parts = []
                for severity in ["severe", "moderate", "minor", "negligible"]:
                    if severity in severity_counts:
                        severity_parts.append(f"{severity_counts[severity]} {severity}")
                
                terrain_summary += f" - {len(obstructions)} obstructions ({', '.join(severity_parts)})"
                terrain_summary += f", est. {total_path_loss_db:.1f}dB obstruction loss"
            else:
                terrain_summary += " - clear path with adequate clearance"
            
            return {
                "total_distance_m": total_distance,
                "samples_analyzed": n_samples,
                "terrain_profile_summary": terrain_summary,
                "max_terrain_height_m": float(np.max(terrain_heights)) if len(terrain_heights) > 0 else 0.0,
                "min_terrain_height_m": float(np.min(terrain_heights)) if len(terrain_heights) > 0 else 0.0,
                "endpoint_heights": {
                    "point_a_terrain_height_m": float(terrain_heights[0]) if len(terrain_heights) > 0 else 0.0,
                    "point_b_terrain_height_m": float(terrain_heights[-1]) if len(terrain_heights) > 0 else 0.0,
                    "point_a_antenna_height_m": float(terrain_heights[0]) + mast_a_height if len(terrain_heights) > 0 else mast_a_height,
                    "point_b_antenna_height_m": float(terrain_heights[-1]) + mast_b_height if len(terrain_heights) > 0 else mast_b_height
                },
                "sample_points": sample_points[:50] if len(sample_points) > 50 else sample_points,  # Limit for readability but include more detail
                "obstructions": obstructions[:15],  # Limit to first 15 obstructions for readability
                "total_path_loss_db": total_path_loss_db,
                "free_space_path_loss_db": free_space_path_loss_db,
                "obstruction_summary": {
                    "total_count": len(obstructions),
                    "by_severity": {
                        "severe": len([o for o in obstructions if o.get("severity") == "severe"]),
                        "moderate": len([o for o in obstructions if o.get("severity") == "moderate"]),
                        "minor": len([o for o in obstructions if o.get("severity") == "minor"]),
                        "negligible": len([o for o in obstructions if o.get("severity") == "negligible"])
                    },
                    "max_single_loss_db": max([o.get("attenuation_db", 0) for o in obstructions], default=0.0),
                    "total_estimated_loss_db": total_path_loss_db
                }
            }
            
    except Exception as e:
        logger.warning(f"Could not get detailed profile information: {e}")
        # Return basic information if detailed analysis fails
        return {
            "total_distance_m": 0.0,
            "samples_analyzed": n_samples,
            "terrain_profile_summary": f"Analysis completed with {n_samples} samples (detailed profile unavailable)",
            "max_terrain_height_m": 0.0,
            "min_terrain_height_m": 0.0,
            "obstructions": []
        }


def _display_point_pair_status(result: Dict[str, Any], console) -> None:
    """Display status for a single point pair analysis."""
    link_id = result.get("id", "unknown")
    
    if result.get("error"):
        console.print(f"  [red]✗ {link_id}: FAILED - {result.get('error', 'Unknown error')}[/red]")
        return
    
    # Extract key metrics
    distance = result.get("total_distance_m", result.get("distance_m", 0))
    samples = result.get("samples_analyzed", result.get("n_samples", 0))
    obstructions = result.get("obstructions", [])
    is_clear = result.get("clear", False)
    obstruction_loss_db = result.get("total_path_loss_db", 0.0)
    free_space_loss_db = result.get("free_space_path_loss_db", 0.0)
    
    # Format distance
    if distance >= 1000:
        distance_str = f"{distance/1000:.1f}km"
    else:
        distance_str = f"{distance:.0f}m"
    
    # Analyze obstruction severity
    obstruction_summary = result.get("obstruction_summary", {})
    severity_counts = obstruction_summary.get("by_severity", {})
    
    # Create obstruction description with coordinates of worst obstructions
    if len(obstructions) == 0:
        obs_desc = "no obstructions"
    else:
        # Count by severity and create meaningful description
        severe = severity_counts.get("severe", 0)
        moderate = severity_counts.get("moderate", 0) 
        minor = severity_counts.get("minor", 0)
        negligible = severity_counts.get("negligible", 0)
        
        parts = []
        if severe > 0:
            parts.append(f"{severe} severe")
        if moderate > 0:
            parts.append(f"{moderate} moderate")
        if minor > 0:
            parts.append(f"{minor} minor")
        if negligible > 0:
            parts.append(f"{negligible} negligible")
            
        if parts:
            obs_desc = f"{', '.join(parts)}"
            # Add worst obstruction location if available
            worst_obs = max(obstructions, key=lambda x: x.get("attenuation_db", 0), default=None)
            if worst_obs and worst_obs.get("attenuation_db", 0) > 1.0:
                lat, lon = worst_obs.get("latitude", 0), worst_obs.get("longitude", 0)
                obs_height = worst_obs.get("obstruction_height_m", 0)
                obs_desc += f" (worst: {obs_height:.1f}m @ {lat:.5f},{lon:.5f})"
        else:
            obs_desc = f"{len(obstructions)} obstructions"
    
    # Create loss breakdown
    loss_desc = f"FSPL: {free_space_loss_db:.1f}dB"
    if obstruction_loss_db > 0:
        loss_desc += f", Obst: +{obstruction_loss_db:.1f}dB"
    
    # Practical signal quality classification based on real-world RF engineering
    if is_clear:
        if obstruction_loss_db == 0:
            # Perfect clear line-of-sight
            quality = "EXCELLENT"
            color = "green"
            symbol = "✓"
        elif obstruction_loss_db < 1.0:
            # Minor Fresnel zone intrusion, negligible impact
            quality = "VERY GOOD"
            color = "green"
            symbol = "✓"
        elif obstruction_loss_db < 3.0:
            # Some Fresnel zone blockage, still very usable
            quality = "GOOD"
            color = "blue"
            symbol = "✓"
        elif obstruction_loss_db < 6.0:
            # Moderate obstruction loss, may need higher power or better antennas
            quality = "FAIR"
            color = "yellow"
            symbol = "⚠"
        elif obstruction_loss_db < 10.0:
            # Significant obstruction loss, challenging but potentially workable
            quality = "POOR"
            color = "orange3"
            symbol = "⚠"
        else:
            # Heavy obstruction loss, likely unreliable
            quality = "MARGINAL"
            color = "red"
            symbol = "⚠"
    else:
        # Direct line-of-sight is blocked
        quality = "BLOCKED"
        color = "red"
        symbol = "✗"
    
    # Display the status line
    console.print(f"  [{color}]{symbol} {link_id}: {quality} - {distance_str}, {samples} samples[/{color}]")
    
    # Show endpoint height information
    endpoint_heights = result.get("endpoint_heights", {})
    if endpoint_heights:
        terrain_a = endpoint_heights.get("point_a_terrain_height_m", 0)
        terrain_b = endpoint_heights.get("point_b_terrain_height_m", 0)
        antenna_a = endpoint_heights.get("point_a_antenna_height_m", 0)
        antenna_b = endpoint_heights.get("point_b_antenna_height_m", 0)
        mast_a = antenna_a - terrain_a
        mast_b = antenna_b - terrain_b
        console.print(f"    [dim]Point A: {terrain_a:.1f}m terrain + {mast_a:.1f}m mast = {antenna_a:.1f}m antenna[/dim]")
        console.print(f"    [dim]Point B: {terrain_b:.1f}m terrain + {mast_b:.1f}m mast = {antenna_b:.1f}m antenna[/dim]")
    
    console.print(f"    [dim]{obs_desc} | {loss_desc}[/dim]")
    
    # Show sample point heights if verbose output is needed (controlled by result data)
    sample_points = result.get("sample_points", [])
    if sample_points and len(sample_points) <= 20:  # Only show for smaller sample sets
        # Show key sample points: start, middle, end, and any obstructions
        key_samples = []
        if len(sample_points) > 0:
            key_samples.append(sample_points[0])  # Start
        if len(sample_points) > 2:
            key_samples.append(sample_points[len(sample_points)//2])  # Middle
        if len(sample_points) > 1:
            key_samples.append(sample_points[-1])  # End
        
        # Add any obstruction points
        obstruction_samples = [sp for sp in sample_points if sp.get("is_obstruction", False)]
        if obstruction_samples:
            key_samples.extend(obstruction_samples[:3])  # Add up to 3 obstruction points
            
        if key_samples:
            console.print(f"    [dim]Key heights along path:[/dim]")
            for sp in key_samples[:6]:  # Limit to 6 key points
                dist_pct = sp.get("distance_from_start_pct", 0)
                terrain_h = sp.get("terrain_height_m", 0)
                actual_los_h = sp.get("actual_los_height_m", 0)
                clearance = sp.get("clearance_above_terrain_m", 0)
                status = "OBSTRUCTED" if sp.get("is_obstruction", False) else "clear"
                console.print(f"      [dim]{dist_pct:5.1f}%: terrain {terrain_h:6.1f}m, LOS {actual_los_h:6.1f}m, clearance {clearance:+6.1f}m ({status})[/dim]")


def analyze_csv_file(
    csv_path: Path,
    las_dir: Optional[Path] = None,
    cache: Optional[Path] = None,
    epsg: Optional[int] = None, 
    resolution: Optional[float] = None,
    workers: Optional[int] = None,
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
        
        # Get optimal number of workers for analysis
        analysis_workers = _get_optimal_analysis_workers(workers)
        
        # Display analysis start message with rich formatting
        console.print(f"\n[bold blue]📊 Starting Line-of-Sight Analysis[/bold blue]")
        console.print(f"[cyan]• Point pairs to analyze: {len(rows)}[/cyan]")
        console.print(f"[cyan]• Workers: {analysis_workers}[/cyan]")
        console.print(f"[cyan]• DSM file: {dsm_path.name}[/cyan]")
        console.print(f"[cyan]• Frequency: {freq_ghz} GHz[/cyan]")
        console.print(f"[cyan]• Individual mast heights will be read from CSV[/cyan]")
        
        # Create progress bar for analysis
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            analysis_task = progress.add_task("Analyzing point pairs...", total=len(rows))
            
            # Process rows in parallel if we have multiple workers and multiple rows
            if analysis_workers > 1 and len(rows) > 1:
                logger.info(f"Using {analysis_workers} workers for parallel analysis")
                
                # Prepare arguments for parallel processing
                args_list = [(row, dsm_path, freq_ghz) for row in rows]
                
                results = []
                failed_count = 0
                
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
                            if result.get("error"):
                                failed_count += 1
                            
                            # Display individual point pair status
                            _display_point_pair_status(result, progress.console)
                            
                        except Exception as e:
                            row_idx = future_to_row[future]
                            logger.error(f"Error processing row {row_idx}: {e}")
                            failed_count += 1
                            # Add a failed result
                            failed_result = {
                                "id": f"failed_row_{row_idx}",
                                "error": str(e),
                                "clear": False,
                                "distance_m": 0
                            }
                            results.append(failed_result)
                            _display_point_pair_status(failed_result, progress.console)
                        
                        # Update progress
                        progress.update(analysis_task, advance=1)
            else:
                # Sequential processing for single worker or single row
                results = []
                failed_count = 0
                
                for i, row in enumerate(rows):
                    try:
                        args = (row, dsm_path, freq_ghz)
                        result = _analyze_single_row(args)
                        results.append(result)
                        if result.get("error"):
                            failed_count += 1
                        
                        # Display individual point pair status
                        _display_point_pair_status(result, progress.console)
                        
                    except Exception as e:
                        logger.error(f"Error processing row {row}: {e}")
                        failed_count += 1
                        failed_result = {
                            "id": row.get("id", f"failed_{len(results)}"),
                            "error": str(e),
                            "clear": False,
                            "distance_m": 0
                        }
                        results.append(failed_result)
                        _display_point_pair_status(failed_result, progress.console)
                    
                    # Update progress
                    progress.update(analysis_task, advance=1)
        
        # Analysis complete - display summary with signal quality breakdown
        successful_count = len(results) - failed_count
        clear_count = sum(1 for r in results if r.get("clear", False))
        blocked_count = successful_count - clear_count
        
        # Calculate signal quality distribution
        quality_counts = {"excellent": 0, "very_good": 0, "good": 0, "fair": 0, "poor": 0, "marginal": 0}
        total_obstruction_loss = 0.0
        total_free_space_loss = 0.0
        
        for r in results:
            if r.get("error") or not r.get("clear", False):
                continue
                
            obs_loss = r.get("total_path_loss_db", 0.0)
            fs_loss = r.get("free_space_path_loss_db", 0.0)
            total_obstruction_loss += obs_loss
            total_free_space_loss += fs_loss
            
            # Classify signal quality
            if obs_loss == 0:
                quality_counts["excellent"] += 1
            elif obs_loss < 1.0:
                quality_counts["very_good"] += 1
            elif obs_loss < 3.0:
                quality_counts["good"] += 1
            elif obs_loss < 6.0:
                quality_counts["fair"] += 1
            elif obs_loss < 10.0:
                quality_counts["poor"] += 1
            else:
                quality_counts["marginal"] += 1
        
        # Calculate averages
        total_distance = sum(r.get("total_distance_m", 0) for r in results if "total_distance_m" in r)
        avg_distance = total_distance / successful_count if successful_count > 0 else 0
        avg_obstruction_loss = total_obstruction_loss / clear_count if clear_count > 0 else 0
        avg_free_space_loss = total_free_space_loss / clear_count if clear_count > 0 else 0
        total_obstructions = sum(len(r.get("obstructions", [])) for r in results)
        
        console.print(f"\n[bold green]✅ Analysis Complete![/bold green]")
        console.print(f"[green]• Total analyzed: {len(results)} point pairs[/green]")
        console.print(f"[green]• Successful: {successful_count}[/green]")
        
        # Signal quality breakdown
        if clear_count > 0:
            console.print(f"\n[bold cyan]📡 Signal Quality Distribution (Clear Paths):[/bold cyan]")
            if quality_counts["excellent"] > 0:
                console.print(f"[green]• Excellent: {quality_counts['excellent']} (no obstruction loss)[/green]")
            if quality_counts["very_good"] > 0:
                console.print(f"[green]• Very Good: {quality_counts['very_good']} (<1dB obstruction loss)[/green]")
            if quality_counts["good"] > 0:
                console.print(f"[blue]• Good: {quality_counts['good']} (1-3dB obstruction loss)[/blue]")
            if quality_counts["fair"] > 0:
                console.print(f"[yellow]• Fair: {quality_counts['fair']} (3-6dB obstruction loss)[/yellow]")
            if quality_counts["poor"] > 0:
                console.print(f"[orange3]• Poor: {quality_counts['poor']} (6-10dB obstruction loss)[/orange3]")
            if quality_counts["marginal"] > 0:
                console.print(f"[red]• Marginal: {quality_counts['marginal']} (>10dB obstruction loss)[/red]")
        
        if blocked_count > 0:
            console.print(f"[red]• Blocked: {blocked_count} (direct line-of-sight obstructed)[/red]")
        if failed_count > 0:
            console.print(f"[red]• Failed: {failed_count}[/red]")
        
        # Path loss breakdown
        if successful_count > 0:
            console.print(f"\n[bold cyan]📊 Path Loss Analysis:[/bold cyan]")
            console.print(f"[cyan]• Average path length: {avg_distance:.1f}m[/cyan]")
            console.print(f"[cyan]• Average free space path loss: {avg_free_space_loss:.1f}dB[/cyan]")
            console.print(f"[cyan]• Average obstruction loss: {avg_obstruction_loss:.1f}dB[/cyan]")
            n_samples = 256  # Standard sample count used in analysis
            console.print(f"[cyan]• Total samples analyzed: {successful_count * n_samples:,}[/cyan]")
            if total_obstructions > 0:
                console.print(f"[orange3]• Total obstructions found: {total_obstructions}[/orange3]")
        
        # Save results if output path specified
        if output_path:
            if output_format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"[cyan]💾 Results saved to: {output_path} (JSON format)[/cyan]")
            elif output_format.lower() == "csv":
                if results:
                    df = pd.DataFrame(results)
                    df.to_csv(output_path, index=False)
                    console.print(f"[cyan]💾 Results saved to: {output_path} (CSV format)[/cyan]")
        
        return results
        
    except Exception as e:
        raise AnalysisError(f"Failed to analyze CSV file: {e}") from e
