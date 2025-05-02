# jpmapper-lidar
```
  _   _   _   _   _   _   _   _  
 / \ / \ / \ / \ / \ / \ / \ / \ 
( j | p | m | a | p | p | e | r )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ 
```

📡  🏛🌲🏗🌲 <<>> 🏢🌲🏚  📡

Julian's Point Mapper

(the LiDAR version)

A python preprocessor which
 * takes .las files
 * makes .tif files (rasterizes)

A python app which
 * takes address or lat/lon for two points
 * takes a broadcast frequency
 * takes a local DSM file in geotiff
 * uses all this to calculate
    * distance
    * elevation
    * terrain
    * buildings and obstructions mid-path
    * fresnel zone obstruction

A python app which
 * takes a folder of .las files
 * compares their data to a defined geographical box
 * deletes .las files which don't contain data in that box
 * keeps .las files which do contain data in that box
 * logs its efforts
---

## jpmapper-rasterize (the preprocessor)
This is an application which is intended for use as a preprocessor before using `jpmapper-lidar`, in case you don't already have a GeoTIFF file for the area in which you want to calculate obstructions.

`jpmapper-rasterize` rasterizes all .las / .laz files in a directory into DSM .tif tiles

The application uses python multiprocessing for parallelism, this operation can take a while on large datasets - remember to set workers option appropriately.

### usage example
```
python3 jpmapper-rasterize.py \
  --lasdir /mnt/lidar_tiles \
  --outdir /mnt/output_dsm \
  --resolution 1.0 \
  --workers 32 \
  --force \
  --cleanup-tiles \
  --cleanup-lidar \
  --debug
```

### options
* `--workers` = number of workers to spawn - start with number of physical CPU cores and reduce if you see swapping or CPU exhaustion
* `--lasdir` = directory where we're fetching .las lidar tiles
* `--outdir` = directory where we're dumping DSM .tif files
* `--resolution` = resolution of the rasterization in meters
* `--force` = overwrites any .tif which exists in the target directory and has the same name (normally we'd skip things with the same name)
* `--cleanup-tiles` = deletes the .tif tiles after merging them into a single file
* `--cleanup-lidar` = deletes the .las files after rasterizing them
* `--no-merge` = doesn't merge the .tif tiles into one, just keeps them tiled
* `--debug` = wtf is going on I need to see inside this thing

### requirements
I've been using CentOS 9 because PDAL has good RPM builds there, and I didn't feel like building from source.

You'll need some packages from epel and the standard repo, in CentOS:
* `yum install epel-release`
* `yum install python3 python3-pip python3-devel gcc geos proj proj-devel`
* `yum install --enablerepo=epel PDAL`

On the Python side, you'll need some from pip too:
* `pip3 install rasterio pyproj geopy numpy matplotlib tqdm`

The application will let you know if it doesn't see some requirements at runtime.

### getting your lidar data
For NYC (this was developed for use in NYC),
* ftp to `ftp://ftp.gis.ny.gov` with anonymous auth
* files you want are in `/elevation/LIDAR` directory
* grab either `NYC_2021` or `NYC_TopoBathymetric2017`, they're large
* place in a directory and point `jpmapper-rasterize` at that directory

...but the process is the same for anywhere else too. You can use any .las data you want, as long as it has the locations for which you're searching. Remember:

* You should include anthropogenic features
* You want a DSM (Digital Surface Model) not a DTM
* This matches how the Fresnel zone clearance logic in `jpmapper-lidar` works: if a building or tree sticks into the Fresnel zone, it should block LOS, so you basically want the first reflection representated in the height data.


## jpmapper-lidar (the main application)

### getting your DSM data
See above, this is why we have a preprocessor - you want to run your .las files through `jpmapper-rasterize.py` to create a GeoTIFF DSM of your area.

### cli usage examples

The kitchen sink:
```
python3 jpmapper-lidar.py \
  --point-a "144 Spencer St, Brooklyn, NY" \
  --override-a 100 \
  --point-b "303 Vernon Ave, Brooklyn, NY" \
  --override-b 100 \
  --dsm /mnt/your-geotiff-dsm.tif \
  --frequency-ghz 5.8
```

Just a couple points:
```
python3 jpmapper-lidar.py \
  --point-a 40.792814,-73.945616 \
  --point-b 40.796872,-73.948744 \
  --freq 60 \
  --dsm geodata/merged_dsm.tif
```

### csv usage examples
format your csv like this:
| point_a_lat | point_a_lon | point_b_lat | point_b_lon | frequency_ghz | override_a | override_b |
|-------------|-------------|-------------|-------------|----------------|------------|------------|
| 40.79       | -73.94      | 40.80       | -73.95      | 5.8            | (optional) | (optional) |

and invoke the program like this:
```
python3 jpmapper-lidar.py --dsm geodata/merged_dsm.tif \
  --csv-input pairs.csv \
  --csv-output results.csv
```

### output examples

A normal looking run:
```
[david@blazes-boylan]# python3 jpmapper-lidar.py --point-a 40.792814,-73.945616 --point-b 40.796872,-73.948744 --freq 5.8 --dsm geodata/merged_dsm.tif
Coordinate (40.792814, -73.945616) maps to pixel (row=8750, col=3618) with elevation 41.83 meters
Coordinate (40.796872, -73.948744) maps to pixel (row=5794, col=1884) with elevation 17.71 meters
=== Link Summary ===
Total distance: 522.27 meters
Point A elevation: 41.83 m
Point B elevation: 17.71 m
Frequency: 5.800 GHz (5800000000 Hz)
First Fresnel zone radius (midpoint): 2.60 meters
Obstruction analysis: 107 obstructed, 4 partial, 82 clear, 7 skipped
Verdict: Obstructed
```

When you specify points out of bounds from your DSM, or have a path which runs out of bounds:
```
[david@blazes-boylan]# python3 jpmapper-lidar.py --point-a "144 Spencer St, Brooklyn, NY" --point-b "303 Vernon Ave, Brooklyn, NY" --dsm geodata/merged_dsm.tif --frequency-ghz 5.8
Coordinate (40.694180, -73.955217) maps to pixel (row=10000, col=0) with elevation -9999.00 meters
Coordinate (40.695935, -73.939720) maps to pixel (row=10000, col=5000) with elevation 27.01 meters
ERROR: All path samples were outside the DSM raster bounds.
DSM covers approximately: lat 40.804826 to 40.791097, lon -73.952141 to -73.943120
Midpoint of path: lat 40.695058, lon -73.947468
Analysis failed: No valid samples were found within the DSM coverage. Please verify your input coordinates.
```

### options
* `--point-a` = Start point (lat,lon or address), required
* `--point-b` = End point (lat,lon or address), required
* `--dsm` = Path to DSM raster or folder, required
* `--frequency-ghz` = Transmission frequency in GHz, required
* `--override-a` = Override elevation at point A (meters), use this optionally if you have an antenna, or if the data doesn't properly represent the point elevation for some reason.
* `--override-b` = Override elevation at point B (meters), use this optionally if you have an antenna, or if the data doesn't properly represent the point elevation for some reason.
* `--csv-input` = bulk input option via csv
* `--csv-output` = if you give me a csv, I'll give you a csv back

## jpmapper-boxfilter (the filter app)
Sometimes, say, you have too many .las files. And those suckers are big, and you have to rasterize them. Before you do, which are useful? Which are the ones with the data you want?

To answer this question, we have the jpmapper-boxfilter. Get a directory of .las files together, point this filter at them, and watch the irrelevant ones depart. All efforts are logged in a csv.

You'll need some more python packages for this, so `pip install laspy shapely tqdm`

The whole NYCMesh (mostly the reason this whole thing exists) network fits in a box defined by these four points:
| Corner | Latitude  | Longitude   |
|--------|-----------|-------------|
| SW     | 40.096269 | -74.945492  |
| NW     | 40.972617 | -74.945492  |
| NE     | 40.972617 | -73.016222  |
| SE     | 40.096269 | -73.016222  |

Since this is a rectangle, we can reduce it to two points for simplicity, `NYC_BBOX = box(-74.945492, 40.096269, -73.016222, 40.972617)`

## usage example & output
