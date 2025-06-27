#!/usr/bin/env python3
"""Test the improved line-of-sight analysis with a small subset."""

import sys
import pandas as pd
import tempfile
from pathlib import Path

def create_test_csv(original_csv_path, output_path, max_pairs=5):
    """Create a small test CSV with just a few point pairs."""
    df = pd.read_csv(original_csv_path)
    # Take first few rows
    test_df = df.head(max_pairs)
    test_df.to_csv(output_path, index=False)
    print(f"Created test CSV with {len(test_df)} point pairs: {output_path}")
    return output_path

def main():
    original_csv = "tests/data/points.csv"
    dsm_path = "D:/dsm_cache.tif"
    
    # Create test CSV with just 3 point pairs
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        test_csv = tmp.name
    
    try:
        create_test_csv(original_csv, test_csv, max_pairs=3)
        
        # Run analysis with the test CSV
        print(f"\n🧪 Testing improved analysis with 3 point pairs...")
        print(f"   DSM: {dsm_path}")
        print(f"   CSV: {test_csv}")
        
        import subprocess
        cmd = [
            sys.executable, "-m", "jpmapper.cli.analyze", 
            "csv", test_csv, 
            "--dsm", dsm_path,
            "--resolution", "1.0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        print("\n📊 Results:")
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
            
        print(f"\nReturn code: {result.returncode}")
        
    finally:
        # Clean up test file
        try:
            Path(test_csv).unlink()
        except:
            pass

if __name__ == "__main__":
    main()
