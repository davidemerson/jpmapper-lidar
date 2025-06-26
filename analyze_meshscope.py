#!/usr/bin/env python3
"""
Analyze the full LiDAR dataset at E:\meshscope to demonstrate metadata-aware rasterization.

This script will:
1. Scan the E:\meshscope directory for LAS files and metadata
2. Generate a comprehensive metadata report
3. Demonstrate enhanced rasterization using the metadata
4. Show how shapefile boundaries improve rasterization reliability
"""

from pathlib import Path
import sys
import json
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def scan_meshscope_directory(meshscope_dir: Path) -> Dict[str, Any]:
    """
    Scan the meshscope directory to understand its structure and contents.
    
    Args:
        meshscope_dir: Path to the E:\meshscope directory
        
    Returns:
        Dictionary with directory analysis
    """
    analysis = {
        'directory': str(meshscope_dir),
        'exists': meshscope_dir.exists(),
        'subdirectories': [],
        'las_files': [],
        'shapefile_sets': [],
        'metadata_files': [],
        'total_size_gb': 0.0
    }
    
    if not meshscope_dir.exists():
        log.warning(f"Directory does not exist: {meshscope_dir}")
        return analysis
    
    print(f"Scanning {meshscope_dir}...")
    
    # Scan for subdirectories
    for item in meshscope_dir.iterdir():
        if item.is_dir():
            analysis['subdirectories'].append(str(item.name))
            
    # Scan for LAS files (recursively)
    las_files = []
    for las_file in meshscope_dir.rglob("*.las"):
        try:
            size_mb = las_file.stat().st_size / (1024 * 1024)
            las_files.append({
                'path': str(las_file),
                'name': las_file.name,
                'size_mb': round(size_mb, 2),
                'relative_path': str(las_file.relative_to(meshscope_dir))
            })
            analysis['total_size_gb'] += size_mb / 1024
        except Exception as e:
            log.warning(f"Could not analyze {las_file}: {e}")
    
    analysis['las_files'] = las_files[:50]  # Limit to first 50 for display
    analysis['total_las_files'] = len(las_files)
    analysis['total_size_gb'] = round(analysis['total_size_gb'], 2)
    
    # Scan for shapefiles
    shapefiles = list(meshscope_dir.rglob("*.shp"))
    for shp in shapefiles:
        shp_info = {
            'path': str(shp),
            'name': shp.name,
            'stem': shp.stem,
            'associated_files': []
        }
        
        # Find associated files
        for ext in ['.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx', '.xml']:
            assoc_file = shp.with_suffix(ext)
            if assoc_file.exists():
                shp_info['associated_files'].append(ext)
        
        analysis['shapefile_sets'].append(shp_info)
    
    # Scan for other metadata files
    metadata_patterns = ['*.prj', '*.xml', '*.txt', '*.pdf']
    for pattern in metadata_patterns:
        for meta_file in meshscope_dir.rglob(pattern):
            if meta_file.suffix not in ['.shp', '.dbf', '.shx']:  # Exclude shapefile components
                analysis['metadata_files'].append({
                    'path': str(meta_file),
                    'name': meta_file.name,
                    'type': meta_file.suffix,
                    'size_kb': round(meta_file.stat().st_size / 1024, 2)
                })
    
    return analysis


def demonstrate_with_sample_files(meshscope_dir: Path, max_files: int = 5) -> None:
    """
    Demonstrate metadata-aware rasterization with a sample of files.
    
    Args:
        meshscope_dir: Path to the meshscope directory
        max_files: Maximum number of files to process for demonstration
    """
    try:
        # Import our enhanced rasterizer
        sys.path.insert(0, str(Path(__file__).parent))
        from jpmapper.io.metadata_raster import MetadataAwareRasterizer, create_metadata_report
        
        print(f"\n=== DEMONSTRATING METADATA-AWARE RASTERIZATION ===")
        
        # Create rasterizer
        rasterizer = MetadataAwareRasterizer()
        
        # Find some LAS files to work with
        las_files = list(meshscope_dir.rglob("*.las"))[:max_files]
        
        if not las_files:
            print("No LAS files found for demonstration")
            return
            
        print(f"Processing {len(las_files)} sample files:")
        
        for i, las_file in enumerate(las_files, 1):
            print(f"\n--- File {i}: {las_file.name} ---")
            print(f"Size: {las_file.stat().st_size / (1024*1024):.1f} MB")
            
            # Find metadata files
            metadata_files = rasterizer.find_metadata_files(las_file)
            print(f"Metadata files found: {list(metadata_files.keys())}")
            
            # Try to get CRS
            crs = rasterizer.get_crs_from_metadata(las_file)
            if crs:
                print(f"CRS from metadata: EPSG:{crs}")
            else:
                print("No CRS found in metadata")
            
            # Try to get tile info
            tile_info = rasterizer.get_tile_info(las_file)
            if tile_info:
                print(f"Tile info available: {len(tile_info)} attributes")
                if 'LAS_ID' in tile_info:
                    print(f"  LAS_ID: {tile_info['LAS_ID']}")
                if 'FILENAME' in tile_info:
                    print(f"  Original filename: {tile_info['FILENAME']}")
                if 'bounds' in tile_info:
                    bounds = tile_info['bounds']
                    width = bounds['maxx'] - bounds['minx']
                    height = bounds['maxy'] - bounds['miny']
                    print(f"  Tile size: {width:.1f} x {height:.1f} units")
            else:
                print("No tile info found")
            
            # Try to get accuracy info
            accuracy_info = rasterizer.get_accuracy_info(las_file)
            if accuracy_info:
                print(f"Accuracy info: {len(accuracy_info)} measurements")
                for key, acc in accuracy_info.items():
                    if isinstance(acc, dict) and 'description' in acc:
                        print(f"  {key}: {acc['description']}")
            else:
                print("No accuracy info found")
                
    except ImportError as e:
        print(f"Could not import enhanced rasterizer: {e}")
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


def analyze_shapefile_coverage(meshscope_dir: Path) -> Dict[str, Any]:
    """
    Analyze how well the shapefiles cover the LAS files in the dataset.
    
    Args:
        meshscope_dir: Path to the meshscope directory
        
    Returns:
        Coverage analysis results
    """
    print(f"\n=== ANALYZING SHAPEFILE COVERAGE ===")
    
    try:
        import geopandas as gpd
        
        # Find all shapefiles
        shapefiles = list(meshscope_dir.rglob("*.shp"))
        print(f"Found {len(shapefiles)} shapefiles")
        
        if not shapefiles:
            return {'error': 'No shapefiles found'}
        
        # Analyze the first (or largest) shapefile
        target_shp = None
        max_size = 0
        
        for shp in shapefiles:
            try:
                size = shp.stat().st_size
                if size > max_size:
                    max_size = size
                    target_shp = shp
            except:
                continue
        
        if not target_shp:
            return {'error': 'No readable shapefiles found'}
        
        print(f"Analyzing shapefile: {target_shp.name}")
        
        # Read the shapefile
        gdf = gpd.read_file(target_shp)
        
        analysis = {
            'shapefile': str(target_shp),
            'feature_count': len(gdf),
            'columns': list(gdf.columns),
            'crs': str(gdf.crs) if gdf.crs else 'Unknown',
            'bounds': list(gdf.total_bounds) if not gdf.empty else None,
            'sample_records': []
        }
        
        # Show some sample records
        for i, row in gdf.head(5).iterrows():
            record = {}
            for col in gdf.columns:
                if col != 'geometry':
                    record[col] = row[col]
            analysis['sample_records'].append(record)
        
        # Look for LAS-related columns
        las_columns = []
        for col in gdf.columns:
            if any(keyword in col.upper() for keyword in ['LAS', 'FILE', 'ID', 'NAME']):
                las_columns.append(col)
        
        analysis['las_related_columns'] = las_columns
        
        # Count how many LAS files we can potentially match
        las_files = list(meshscope_dir.rglob("*.las"))
        las_stems = {las.stem for las in las_files}
        
        matched_count = 0
        for col in las_columns:
            for value in gdf[col].dropna():
                value_str = str(value)
                # Remove .las extension if present
                clean_value = value_str.replace('.las', '').replace('.LAS', '')
                if clean_value in las_stems:
                    matched_count += 1
                    break
        
        analysis['potential_matches'] = matched_count
        analysis['total_las_files'] = len(las_files)
        analysis['match_percentage'] = round(matched_count / len(las_files) * 100, 1) if las_files else 0
        
        return analysis
        
    except ImportError:
        return {'error': 'geopandas not available for shapefile analysis'}
    except Exception as e:
        return {'error': f'Error analyzing shapefile: {e}'}


def main():
    """Main function to analyze the meshscope dataset."""
    meshscope_dir = Path("E:/meshscope")
    
    print("=== MESHSCOPE LIDAR DATASET ANALYSIS ===")
    print(f"Target directory: {meshscope_dir}")
    
    # 1. Basic directory scan
    print("\n1. SCANNING DIRECTORY STRUCTURE...")
    analysis = scan_meshscope_directory(meshscope_dir)
    
    print(f"Directory exists: {analysis['exists']}")
    if not analysis['exists']:
        print("Cannot proceed - directory not found")
        return
    
    print(f"Subdirectories: {len(analysis['subdirectories'])}")
    if analysis['subdirectories']:
        print(f"  Examples: {analysis['subdirectories'][:5]}")
    
    print(f"Total LAS files found: {analysis['total_las_files']}")
    print(f"Total dataset size: {analysis['total_size_gb']} GB")
    
    print(f"Shapefile sets: {len(analysis['shapefile_sets'])}")
    for shp in analysis['shapefile_sets'][:3]:  # Show first 3
        print(f"  {shp['name']}: {shp['associated_files']}")
    
    print(f"Other metadata files: {len(analysis['metadata_files'])}")
    for meta in analysis['metadata_files'][:5]:  # Show first 5
        print(f"  {meta['name']} ({meta['size_kb']} KB)")
    
    # 2. Analyze shapefile coverage
    shapefile_analysis = analyze_shapefile_coverage(meshscope_dir)
    if 'error' not in shapefile_analysis:
        print(f"Shapefile features: {shapefile_analysis['feature_count']}")
        print(f"CRS: {shapefile_analysis['crs']}")
        print(f"LAS-related columns: {shapefile_analysis['las_related_columns']}")
        print(f"Potential matches: {shapefile_analysis['potential_matches']}/{shapefile_analysis['total_las_files']} ({shapefile_analysis['match_percentage']}%)")
    else:
        print(f"Shapefile analysis error: {shapefile_analysis['error']}")
    
    # 3. Demonstrate with sample files
    if analysis['total_las_files'] > 0:
        demonstrate_with_sample_files(meshscope_dir, max_files=3)
    
    # 4. Save detailed analysis
    output_file = Path("meshscope_analysis.json")
    full_analysis = {
        'directory_scan': analysis,
        'shapefile_analysis': shapefile_analysis,
        'timestamp': str(Path(__file__).stat().st_mtime)
    }
    
    with open(output_file, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")
    
    print("\n=== RECOMMENDATIONS ===")
    print("Based on this analysis, the metadata files can help ensure:")
    print("1. Correct CRS/projection for each LAS file")
    print("2. Proper tile boundaries for rasterization")
    print("3. Quality metrics for resolution selection")
    print("4. Spatial indexing for efficient processing")
    print("\nUse the enhanced rasterization functions to leverage this metadata!")


if __name__ == "__main__":
    main()
