#!/usr/bin/env python3
"""
Demonstrate the benefits of metadata-aware rasterization with the meshscope dataset.

This script shows the difference between standard rasterization and metadata-enhanced
rasterization, highlighting how metadata files improve reliability and accuracy.
"""

from pathlib import Path
import sys
import tempfile
import time
import json

def compare_rasterization_methods(las_file: Path, output_dir: Path):
    """
    Compare standard vs metadata-enhanced rasterization for a single file.
    
    Args:
        las_file: LAS file to process
        output_dir: Directory for outputs
    """
    print(f"\n=== COMPARING RASTERIZATION METHODS FOR {las_file.name} ===")
    
    # File info
    size_mb = las_file.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    
    results = {}
    
    # 1. Standard rasterization
    print("\n1. STANDARD RASTERIZATION:")
    try:
        from jpmapper.api.raster import rasterize_tile
        
        standard_output = output_dir / f"{las_file.stem}_standard.tif"
        start_time = time.time()
        
        standard_result = rasterize_tile(
            las_file,
            standard_output,
            epsg=6539,  # We have to guess/hardcode this
            resolution=0.1
        )
        
        standard_time = time.time() - start_time
        
        results['standard'] = {
            'output_file': str(standard_result),
            'exists': standard_result.exists(),
            'size_mb': standard_result.stat().st_size / (1024 * 1024) if standard_result.exists() else 0,
            'processing_time': standard_time,
            'epsg': 6539,  # Hardcoded
            'resolution': 0.1,  # Hardcoded
            'crs_source': 'hardcoded',
            'metadata_used': False
        }
        
        print(f"✓ Success: {standard_result.name}")
        print(f"  Processing time: {standard_time:.2f}s")
        print(f"  Output size: {results['standard']['size_mb']:.1f} MB")
        print(f"  CRS: EPSG:{results['standard']['epsg']} (hardcoded)")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['standard'] = {'error': str(e)}
    
    # 2. Metadata-enhanced rasterization
    print("\n2. METADATA-ENHANCED RASTERIZATION:")
    try:
        # Add project root to path
        sys.path.insert(0, str(Path(__file__).parent))
        from jpmapper.api.enhanced_raster import rasterize_tile_with_metadata
        
        enhanced_output = output_dir / f"{las_file.stem}_enhanced.tif"
        start_time = time.time()
        
        enhanced_result, metadata_info = rasterize_tile_with_metadata(
            las_file,
            enhanced_output,
            use_metadata=True,
            auto_adjust_resolution=True
        )
        
        enhanced_time = time.time() - start_time
        
        results['enhanced'] = {
            'output_file': str(enhanced_result),
            'exists': enhanced_result.exists(),
            'size_mb': enhanced_result.stat().st_size / (1024 * 1024) if enhanced_result.exists() else 0,
            'processing_time': enhanced_time,
            'metadata_info': metadata_info,
            'metadata_used': metadata_info.get('metadata_enhanced', False)
        }
        
        print(f"✓ Success: {enhanced_result.name}")
        print(f"  Processing time: {enhanced_time:.2f}s")
        print(f"  Output size: {results['enhanced']['size_mb']:.1f} MB")
        print(f"  CRS source: {metadata_info.get('crs_source', 'unknown')}")
        print(f"  EPSG: {metadata_info.get('used_epsg', 'unknown')}")
        print(f"  Resolution: {metadata_info.get('resolution', 'unknown')} m")
        print(f"  Metadata enhanced: {metadata_info.get('metadata_enhanced', False)}")
        
        if 'tile_info' in metadata_info:
            tile_info = metadata_info['tile_info']
            print(f"  Tile ID: {tile_info.get('LAS_ID', 'unknown')}")
            if 'bounds' in tile_info:
                bounds = tile_info['bounds']
                print(f"  Tile bounds: {bounds['minx']:.1f}, {bounds['miny']:.1f} to {bounds['maxx']:.1f}, {bounds['maxy']:.1f}")
        
        if 'accuracy_info' in metadata_info:
            print(f"  Accuracy data available: {list(metadata_info['accuracy_info'].keys())}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['enhanced'] = {'error': str(e)}
    
    # 3. Comparison summary
    print("\n3. COMPARISON SUMMARY:")
    
    if 'error' not in results.get('standard', {}) and 'error' not in results.get('enhanced', {}):
        print("Both methods succeeded!")
        
        # Compare processing times
        time_diff = results['enhanced']['processing_time'] - results['standard']['processing_time']
        if time_diff > 0:
            print(f"  Enhanced method took {time_diff:.2f}s longer (metadata processing overhead)")
        else:
            print(f"  Enhanced method was {abs(time_diff):.2f}s faster")
        
        # Compare output sizes
        size_diff = results['enhanced']['size_mb'] - results['standard']['size_mb']
        if abs(size_diff) > 0.1:
            print(f"  Output size difference: {size_diff:+.1f} MB")
        else:
            print("  Output sizes are similar")
        
        # Compare metadata usage
        if results['enhanced']['metadata_used']:
            print("  ✓ Enhanced method used metadata for improved accuracy")
            enhanced_meta = results['enhanced']['metadata_info']
            if enhanced_meta.get('crs_source') != 'hardcoded':
                print(f"    - CRS auto-detected from {enhanced_meta.get('crs_source', 'metadata')}")
            if 'tile_info' in enhanced_meta:
                print("    - Used tile boundary information")
            if 'accuracy_info' in enhanced_meta:
                print("    - Used accuracy data for quality assessment")
        else:
            print("  ⚠ Enhanced method fell back to standard processing")
    
    else:
        print("One or both methods failed - check error messages above")
    
    return results


def demonstrate_batch_processing(las_dir: Path, output_dir: Path, max_files: int = 3):
    """
    Demonstrate batch processing with metadata enhancement.
    
    Args:
        las_dir: Directory containing LAS files
        output_dir: Directory for outputs
        max_files: Maximum number of files to process
    """
    print(f"\n=== BATCH PROCESSING DEMONSTRATION ===")
    
    las_files = list(las_dir.glob("*.las"))[:max_files]
    
    if not las_files:
        print("No LAS files found for batch processing")
        return
    
    print(f"Processing {len(las_files)} files with metadata enhancement...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from jpmapper.api.enhanced_raster import batch_rasterize_with_metadata, generate_processing_report
        
        # Batch process with metadata enhancement
        start_time = time.time()
        
        results = batch_rasterize_with_metadata(
            las_files,
            output_dir / "batch_enhanced",
            use_metadata=True,
            auto_adjust_resolution=True
        )
        
        batch_time = time.time() - start_time
        
        print(f"Batch processing completed in {batch_time:.2f}s")
        print(f"Successfully processed: {len(results)} files")
        
        # Generate report
        report = generate_processing_report(
            results,
            output_dir / "batch_processing_report.json"
        )
        
        print(f"\nBATCH PROCESSING SUMMARY:")
        summary = report['summary']
        print(f"  Total files: {summary['total_files']}")
        print(f"  Metadata enhanced: {summary['metadata_enhanced']}")
        print(f"  Resolution optimized: {summary['resolution_optimized']}")
        print(f"  Quality issues: {summary['quality_issues']}")
        
        if summary['crs_sources']:
            print(f"  CRS sources: {dict(summary['crs_sources'])}")
        
        if summary['resolution_distribution']:
            print(f"  Resolution distribution: {dict(summary['resolution_distribution'])}")
        
        if report['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        return results, report
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main demonstration function."""
    print("=== METADATA-AWARE RASTERIZATION DEMONSTRATION ===")
    
    # Use the meshscope dataset
    meshscope_dir = Path("E:/meshscope")
    
    if not meshscope_dir.exists():
        print(f"Dataset not found at {meshscope_dir}")
        print("Please update the path to your LiDAR dataset")
        return
    
    # Create output directory
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Find some LAS files to demonstrate with
    las_files = list(meshscope_dir.glob("*.las"))
    
    if not las_files:
        print("No LAS files found in the dataset")
        return
    
    print(f"Found {len(las_files)} LAS files in dataset")
    
    # 1. Single file comparison
    test_file = las_files[0]  # Use the first file
    print(f"Using test file: {test_file.name}")
    
    comparison_results = compare_rasterization_methods(test_file, output_dir)
    
    # Save comparison results
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    # 2. Batch processing demonstration
    batch_results, batch_report = demonstrate_batch_processing(
        meshscope_dir, 
        output_dir, 
        max_files=3
    )
    
    print(f"\n=== SUMMARY ===")
    print("This demonstration shows how metadata files improve rasterization by:")
    print("1. Auto-detecting correct CRS/projection from .prj files")
    print("2. Using tile boundary information from shapefiles")
    print("3. Leveraging accuracy data for quality assessment")
    print("4. Providing detailed processing reports")
    print(f"\nResults saved to: {output_dir}")
    
    # Show the key benefits
    if comparison_results and 'enhanced' in comparison_results:
        enhanced_info = comparison_results['enhanced'].get('metadata_info', {})
        if enhanced_info.get('metadata_enhanced'):
            print(f"\n✓ Metadata enhancement was successful!")
            print(f"  CRS source: {enhanced_info.get('crs_source', 'unknown')}")
            print(f"  Used tile info: {'tile_info' in enhanced_info}")
            print(f"  Used accuracy data: {'accuracy_info' in enhanced_info}")
        else:
            print(f"\n⚠ Metadata enhancement fell back to standard processing")
    
    print(f"\nFor the full dataset of {len(las_files)} files, metadata-aware rasterization would provide:")
    print("- Consistent CRS handling across all tiles")
    print("- Proper spatial alignment using tile boundaries") 
    print("- Quality-appropriate resolution selection")
    print("- Comprehensive processing reports and error handling")


if __name__ == "__main__":
    main()
