#!/usr/bin/env python3
"""Test script to run analysis and check output"""

import sys
import subprocess
from pathlib import Path

def test_analysis():
    """Test running the analysis command"""
    print("Testing analysis command...")
    
    # Check if test files exist
    test_csv = Path("tests/data/test_points.csv")
    output_tif = Path("output.tif")
    
    if not test_csv.exists():
        print(f"❌ Test CSV not found: {test_csv}")
        return False
        
    if not output_tif.exists():
        print(f"❌ Output TIF not found: {output_tif}")
        return False
        
    print(f"✅ Test CSV exists: {test_csv}")
    print(f"✅ Output TIF exists: {output_tif}")
    
    # Run the analysis command
    cmd = [sys.executable, "-m", "jpmapper.cli.main", "analyze", 
           str(test_csv), str(output_tif), "--map", "--verbose"]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check for output files
        map_files = list(Path(".").glob("*analysis_map*.png"))
        if map_files:
            print(f"✅ Found analysis map files: {map_files}")
        else:
            print("❌ No analysis map files found")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

if __name__ == "__main__":
    test_analysis()
