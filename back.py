import openeo
import json
from pathlib import Path
import folium

connection = openeo.connect(
    "openeo.dataspace.copernicus.eu"
).authenticate_oidc()

# Load geographical area of interest
def read_json(filename: str) -> dict:
    with open(filename) as input:
        field = json.load(input)
    return field

date = ["2023-06-01", "2023-10-30"]
aoi = read_json("Netherlands_polygon.geojson")

# Create a map
m = folium.Map([52.2, 5], zoom_start=7)
folium.GeoJson(aoi).add_to(m)


# Load Sentinel-3 SLSTR L2 LST data for the Netherlands
lst = connection.load_collection(
    "SENTINEL3_SLSTR_L2_LST",
    temporal_extent=date,
    spatial_extent=aoi,
    bands=["LST"],
)

# Apply cloud masking
mask = connection.load_collection(
    "SENTINEL3_SLSTR_L2_LST",
    temporal_extent=date,
    spatial_extent=aoi,
    bands=["confidence_in"],
)
mask = mask >= 16384
lst.mask(mask)

# User Defined Function for heatwave detection
udf = openeo.UDF(
    """
import xarray
import numpy as np
from openeo.udf import inspect

def apply_datacube(cube: xarray.DataArray, context: dict) -> xarray.DataArray:
    array = cube.values
    inspect(data=[array.shape], message = "Array dimensions")
    res_arr=np.zeros(array.shape)
    for i in range(array.shape[0]-4):
        ar_sub=np.take(array,  range(i, i+5), axis=0)
        res_arr[i]=(np.all(ar_sub>295,axis=0)) & (np.nansum(ar_sub>300,axis=0)>2)
    return xarray.DataArray(res_arr, dims=cube.dims, coords=cube.coords)
"""
)

# Apply UDF to the data
heatwave_loc = lst.apply_dimension(process=udf, dimension="t")
heatwave_loc = heatwave_loc.reduce_dimension(reducer="sum", dimension="t")

job_options = {
    "executor-memory": "3G",
    "executor-memoryOverhead": "4G",
    "executor-cores": "2",
}

heatwave_job = heatwave_loc.execute_batch(
    title="Heatwave Locations in the Netherlands",
    outputfile="Heatwave_NL.nc"
)

import matplotlib
import xarray as xr
import numpy as np

heatwave = xr.load_dataset("Heatwave_NL.nc")
data = heatwave[["LST"]].to_array(dim="bands")[0]
data.values[data == 0] = np.nan

# Interactive Plot using Folium
lon, lat = np.meshgrid(data.x.values.astype(np.float64), data.y.values.astype(np.float64))
cm = matplotlib.colormaps.get_cmap('hot_r')
colored_data = cm(data/10)

m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=8)
folium.raster_layers.ImageOverlay(colored_data,
                     [[lat.min(), lon.min()], [lat.max(), lon.max()]],
                     mercator_project=True,
                     opacity=0.5).add_to(m)
m

# Static Plot using Cartopy
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

axes = plt.axes(projection=ccrs.PlateCarree())
axes.coastlines()
axes.add_feature(cfeature.BORDERS, linestyle=':')
data.plot.imshow(vmin=0, vmax=10, ax=axes, cmap="hot_r")
axes.set_title("# of Days with Heatwave in 2023")



###########################################################


import openeo
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

# Define spatial extent for Ischia Island
spatial_extent = {
    "west": 13.882567409197492,
    "south": 40.7150627793427,
    "east": 13.928593792166282,
    "north": 40.747050251559216,
}

# Load pre-event Sentinel-2 data (before landslide)
s2pre = connection.load_collection(
    "SENTINEL2_L2A",
    temporal_extent=["2022-08-25", "2022-11-25"],
    spatial_extent=spatial_extent,
    bands=["B04", "B08"],
)

# Calculate pre-NDVI
prendvi = s2pre.ndvi().mean_time()

# Load post-event Sentinel-2 data (after landslide)
s2post = connection.load_collection(
    "SENTINEL2_L2A",
    temporal_extent=["2022-11-26", "2022-12-25"],
    spatial_extent=spatial_extent,
    bands=["B04", "B08"],
)

# Calculate post-NDVI
postndvi = s2post.ndvi().mean_time()

# Calculate NDVI difference
diff = postndvi - prendvi
diff.download("NDVIDiff.tiff")

# Load the calculated NDVI difference data
img = rasterio.open("NDVIDiff.tiff")
value = img.read(1)
cmap = matplotlib.colors.ListedColormap(["black", "red"])

# Plot the result
fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
im = show(
    ((value < -0.48) & (value > -1)),
    vmin=0,
    vmax=1,
    cmap=cmap,
    transform=img.transform,
    ax=ax,
)

# Set plot labels and legend
values = ["Absence", "Presence"]
colors = ["black", "red"]
ax.set_title("Detected Landslide Area")
ax.set_xlabel("X Coordinates")
ax.set_ylabel("Y Coordinates")
patches = [
    mpatches.Patch(color=colors[i], label="Landslide {l}".format(l=values[i]))
    for i in range(len(values))
]
fig.legend(handles=patches, bbox_to_anchor=(0.83, 1.03), loc=1)


################################################

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

# Define area of interest (Belgium) and time period
extent = {"west": 2.146728, "south": 49.446978, "east": 6.497314, "north": 51.651837}
time = ["2020-01-01", "2021-01-01"]

# Fetch Sentinel-5P data for multiple atmospheric bands
def fetch_collection(bands, time, extent) -> list:
    cubes = []
    for band in bands:
        datacube = connection.load_collection(
            "SENTINEL5P_L2",
            temporal_extent=time,
            spatial_extent=extent,
            bands=[band],
        )
        cubes.append(datacube)
    return cubes

atmospheric_data = fetch_collection(
    bands=["NO2", "SO2", "CO", "O3", "CH4", "AER_AI"],
    time=time,
    extent=extent,
)

import matplotlib.pyplot as plt
import numpy as np

# Example of visualizing the NO2 concentration
no2 = atmospheric_data[0]
no2_mean = no2.mean_time()
no2_data = no2_mean.values

# Visualizing the NO2 concentration
plt.imshow(no2_data, cmap="Blues")
plt.colorbar(label="NO2 Concentration (mol/m^2)")
plt.title("NO2 Concentration in Belgium (2020)")
plt.show()
