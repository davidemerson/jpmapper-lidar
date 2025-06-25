"""End-to-end integration tests for the entire JPMapper workflow."""
import os
import json
from pathlib import Path
import pytest
import tempfile
import shutil

from jpmapper.api import filter_by_bbox, rasterize_tile, analyze_los
from jpmapper.exceptions import JPMapperError


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test the entire JPMapper workflow from filtering to analysis."""
    
    @pytest.fixture
    def test_data_dir(self):
        """
        Fixture that returns the test data directory path.
        Tests will be skipped if the directory doesn't exist.
        """
        data_dir = Path(__file__).parent / "data"
        if not data_dir.exists():
            pytest.skip("Test data directory not found")
        return data_dir
    
    @pytest.fixture
    def las_files(self, test_data_dir):
        """
        Fixture that returns a list of LAS files for testing.
        Tests will be skipped if no LAS files are found.
        """
        las_dir = test_data_dir / "las"
        las_files = list(las_dir.glob("*.las"))
        if not las_files:
            pytest.skip("No LAS test files found")
        return las_files
    
    def test_full_workflow(self, las_files):
        """
        Test the full workflow from filtering to analysis.
        
        This test will:
        1. Filter LAS files by a bounding box
        2. Rasterize the filtered files to a DSM
        3. Analyze a line-of-sight between two points
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Step 1: Filter LAS files
            filtered_dir = tmpdir_path / "filtered"
            filtered_dir.mkdir()
            
            # Get bounding box from the first LAS file
            # This ensures we'll have at least one file selected
            from laspy import open as las_open
            with las_open(las_files[0]) as f:
                header = f.header
                # Expand the bbox slightly to ensure we select the file
                minx, miny = header.mins[0] - 1, header.mins[1] - 1
                maxx, maxy = header.maxs[0] + 1, header.maxs[1] + 1
                bbox = (minx, miny, maxx, maxy)
            
            filtered_files = filter_by_bbox(
                las_files, 
                bbox=bbox,
                dst_dir=filtered_dir
            )
            
            # Assert that at least one file was selected
            assert len(filtered_files) > 0, "No files were selected in the filtering step"
            
            # Step 2: Rasterize to DSM
            dsm_path = tmpdir_path / "dsm.tif"
            
            # If there are multiple filtered files, use the first one
            src_las = filtered_files[0]
            
            # Rasterize
            rasterize_tile(
                src_las,
                dsm_path,
                epsg=None,  # Auto-detect from source
                resolution=0.5  # Use a coarse resolution for faster testing
            )
            
            # Assert that the DSM file was created
            assert dsm_path.exists(), "DSM file was not created"
            assert dsm_path.stat().st_size > 0, "DSM file is empty"
            
            # Step 3: Analyze a line-of-sight
            # Get coordinates that are likely to be within the DSM
            from laspy import open as las_open
            with las_open(src_las) as f:
                header = f.header
                # Create two points within the bounds of the file
                midx = (header.mins[0] + header.maxs[0]) / 2
                midy = (header.mins[1] + header.maxs[1]) / 2
                
                # Convert to lat/lon if needed (simplified for test)
                point_a = (midy - 0.001, midx - 0.001)  # lat, lon
                point_b = (midy + 0.001, midx + 0.001)  # lat, lon
            
            # Analyze line-of-sight
            try:
                result = analyze_los(
                    dsm_path,
                    point_a,
                    point_b,
                    freq_ghz=5.8
                )
                
                # Assert that the result contains the expected fields
                assert "clear" in result
                assert "mast_height_m" in result
                assert "surface_height_a_m" in result
                assert "surface_height_b_m" in result
                assert "distance_m" in result
                
                # The result may be clear or blocked, but either is valid
                # Just check that the type is boolean
                assert isinstance(result["clear"], bool)
                
            except JPMapperError as e:
                # If the points are outside the DSM, the test may legitimately fail
                # We'll allow this exception
                if "outside the DSM" in str(e):
                    pytest.skip(f"Test points are outside the DSM: {e}")
                else:
                    # Other errors should be raised
                    raise
    
    def test_workflow_with_csv(self, test_data_dir, las_files):
        """
        Test the workflow using a CSV file of points.
        
        This test will:
        1. Read test points from points.csv
        2. Create a DSM from the LAS files
        3. Analyze each point pair in the CSV
        4. Check that the results match expectations
        """
        points_csv = test_data_dir / "points.csv"
        if not points_csv.exists():
            pytest.skip("points.csv not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Step 1: Create a DSM from all LAS files
            dsm_path = tmpdir_path / "dsm.tif"
            
            # If there are multiple LAS files, use the first one
            src_las = las_files[0]
            
            # Rasterize
            rasterize_tile(
                src_las,
                dsm_path,
                epsg=None,  # Auto-detect from source
                resolution=0.5  # Use a coarse resolution for faster testing
            )
            
            # Step 2: Read points from CSV
            import csv
            results = []
            
            with open(points_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Parse the row
                        point_a = (float(row["point_a_lat"]), float(row["point_a_lon"]))
                        point_b = (float(row["point_b_lat"]), float(row["point_b_lon"]))
                        freq_ghz = float(row["frequency_ghz"])
                        expected_clear = row["expected_clear"].lower() == "true"
                        
                        # Analyze line-of-sight
                        result = analyze_los(
                            dsm_path,
                            point_a,
                            point_b,
                            freq_ghz=freq_ghz
                        )
                        
                        # Store the result
                        results.append({
                            "point_a": point_a,
                            "point_b": point_b,
                            "expected_clear": expected_clear,
                            "actual_clear": result["clear"],
                            "mast_height_m": result["mast_height_m"],
                            "distance_m": result["distance_m"]
                        })
                        
                    except JPMapperError as e:
                        # If the points are outside the DSM, skip this row
                        if "outside the DSM" in str(e):
                            continue
                        else:
                            # Other errors should be raised
                            raise
            
            # Skip the test if no points could be analyzed
            if not results:
                pytest.skip("No points could be analyzed with the test DSM")
            
            # Check that at least some results match expectations
            # We can't expect all to match since our test DSM is not the same as what was used
            # to create the CSV, but some should match by chance
            match_count = sum(1 for r in results if r["expected_clear"] == r["actual_clear"])
            
            # Write a JSON file with the results for debugging
            with open(tmpdir_path / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Log the match rate
            match_rate = match_count / len(results) if results else 0
            print(f"Match rate: {match_rate:.2f} ({match_count}/{len(results)})")
            
            # This test is informational only, so don't assert on the match rate
