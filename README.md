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
---

## jpmapper-rasterize (the preprocessor)
This is an application which is intended for use as a preprocessor before using jpmapper-lidar, in case you don't already have a GeoTIFF file for the area in which you want to calculate obstructions.

jpmapper-rasterize rasterizes all .las / .laz files in a directory into DSM .tif tiles

The application uses python multiprocessing for parallelism, this operation can take a while on large datasets.

Once the DSM tiles are created, they are merged DSM using gdal_merge.py (optionally disabled).

### usage example
```
python3 jpmapper-rasterize.py \
  --lasdir /mnt/lidar_tiles \
  --outdir /mnt/output_dsm \
  --resolution 1.0 \
  --workers 32 \
  --force \
  --cleanup-tiles \
  --cleanup-lidar
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

### requirements
You'll need python3.x and tqdm, so `pip install tqdm`

You'll also want to install pdal and gdal-bin, in debian: `sudo apt install pdal gdal-bin`

### getting your lidar data
For NYC (this was developed for use in NYC),
* ftp to `ftp://ftp.gis.ny.gov/elevation/LIDAR` with anonymous auth
* grab either NYC_2021 or NYC_TopoBathymetric2017, they're large
* place in a directory and point jpmapper-rasterize at that directory

...but the process is the same for anywhere else too. You can use any .las data you want, as long as it has the locations for which you're searching. Remember:

* You should include anthropogenic features
* You want a DSM (Digital Surface Model) not a DTM
* This matches how the Fresnel zone clearance logic in jpmapper-lidar works: if a building or tree sticks into the Fresnel zone, it should block LOS, so you basically want the first reflection representated in the height data.


## jpmapper-lidar (the main application)
Usage Examples:
  ./jpmapper -addr1 "Empire State Building, NYC" -addr2 "One World Trade Center, NYC" -freq 5800
  ./jpmapper -lat1 40.7484 -lon1 -73.9857 -lat2 40.7127 -lon2 -74.0134 -freq 2400 -debug
  ./jpmapper -addr1 "144 Spencer St, Brooklyn, NY" -addr2 "303 Vernon Ave, Brooklyn, NY" -freq 5800

Notes:
        * addresses and place names will be resolved to lat/lon
        * freq is in MHz
        * debug shows obstruction work so you can troubleshoot buildings
