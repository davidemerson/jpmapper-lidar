# Test Data Directory

This directory contains test data files used by the JPMapper test suite.

## Directory Structure

- `las/`: Place LAS/LAZ files here for testing LiDAR operations
- `points.csv`: Test points for line-of-sight analysis

## Adding Test Data

To run the integration tests, you'll need to add some test LAS files:

1. Place one or more LAS/LAZ files in the `las/` directory
2. The files should be small (preferably < 5MB) to keep the repository size manageable
3. Ideally, the files should cover an area with varying elevation

## Test Data Format

### points.csv

The `points.csv` file contains test points for line-of-sight analysis with the following columns:

- `point_a_lat`: Latitude of point A
- `point_a_lon`: Longitude of point A
- `point_b_lat`: Latitude of point B
- `point_b_lon`: Longitude of point B
- `frequency_ghz`: Frequency in GHz for Fresnel zone calculation
- `expected_clear`: Expected result of the line-of-sight analysis ("true" or "false")

## Test Skipping

Tests that require data files are designed to skip automatically if the required files are not found. This allows the basic test suite to run without any test data, while integration tests can be run when data is available.
