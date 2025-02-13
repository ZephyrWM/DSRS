import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import pandas as pd

# ===============================
# 1. Data Directory and File Patterns
# ===============================
data_dir = r"D:\DSRS\temp\RG_ArgoClim_Temperature_2019"

pattern_monthly = re.compile(r"RG_ArgoClim_(\d{6})_\d{4}\.nc")
pattern_annual = re.compile(r"RG_ArgoClim_Temperature_(\d{4})\.nc")

lat_range = (-65, -40)  # Latitude range (-65°S to -40°S)

# ===============================
# 2. Read and Concatenate Dataset
# ===============================
ds_list = []

for filename in tqdm(sorted(os.listdir(data_dir))):
    file_path = os.path.join(data_dir, filename)
    
    if pattern_annual.match(filename):
        ds = xr.open_dataset(file_path, decode_times=False)
        ds = ds.sel(LATITUDE=slice(lat_range[0], lat_range[1]))
        
        time_values = ds["TIME"].values
        start_date = pd.Timestamp("2004-01-01")
        time_decoded = [start_date + pd.DateOffset(months=int(t)) for t in time_values]
        ds = ds.assign_coords(TIME=pd.to_datetime(time_decoded))
        
        if "LONGITUDE" in ds.dims:
            ds = ds.mean(dim="LONGITUDE")
        
        ds_list.append(ds)
        ds.close()

    elif pattern_monthly.match(filename):
        match = pattern_monthly.search(filename)
        if match:
            month_str = match.group(1)
            year = int(month_str[:4])
            month = int(month_str[4:])
            time_point = pd.Timestamp(f"{year}-{month:02d}")
            
            ds = xr.open_dataset(file_path, decode_times=False)
            ds = ds.sel(LATITUDE=slice(lat_range[0], lat_range[1]))
            
            if "TIME" not in ds.dims:
                ds = ds.expand_dims("TIME")
            ds = ds.assign_coords(TIME=[time_point])
            
            if "LONGITUDE" in ds.dims:
                ds = ds.mean(dim="LONGITUDE")
            
            ds_list.append(ds)
            ds.close()

ds_all = xr.concat(ds_list, dim="TIME")
ds_all = ds_all.sortby("TIME")

# ===============================
# 3. Convert TIME to Numeric Years
# ===============================
time_numeric = ds_all["TIME"].dt.year + (ds_all["TIME"].dt.month - 1) / 12.0
ds_all = ds_all.assign_coords(time_numeric=time_numeric)
ds_all = ds_all.swap_dims({"TIME": "time_numeric"})

# ===============================
# 4. Calculate Linear Temperature Trend (°C/yr)
# ===============================
trend_fit = ds_all["ARGO_TEMPERATURE_ANOMALY"].polyfit(dim="time_numeric", deg=1)
slope = trend_fit.polyfit_coefficients.sel(degree=1)

# ===============================
# 5. Ensure `slope` and `ARGO_TEMPERATURE_MEAN` are 2D
# ===============================
if "time_numeric" in slope.dims:
    slope_2d = slope.mean(dim="time_numeric")
else:
    slope_2d = slope

mean_temperature = ds_all["ARGO_TEMPERATURE_MEAN"]
if "time_numeric" in mean_temperature.dims:
    mean_temperature_2d = mean_temperature.mean(dim="time_numeric")
else:
    mean_temperature_2d = mean_temperature

# ===============================
# 6. Plot Linear Trend with Mean Temperature Contours
# ===============================
plt.figure(figsize=(12, 8))

# Plot the linear temperature trend
cf = plt.contourf(slope_2d["LATITUDE"], slope_2d["PRESSURE"], slope_2d,
                  levels=20, cmap="Reds", extend='both')
cbar = plt.colorbar(cf)
cbar.set_label("Temperature Trend (°C/yr)")

# Overlay mean temperature contours
contour_levels = np.arange(-2, 30, 1)  # Adjust the range if necessary
mean_contours = plt.contour(mean_temperature_2d["LATITUDE"], mean_temperature_2d["PRESSURE"],
                           mean_temperature_2d, levels=contour_levels, colors='black', linewidths=0.5)
plt.clabel(mean_contours, inline=True, fontsize=8, fmt="%.1f")

# Label axes and title
plt.xlabel("Latitude")
plt.ylabel("Pressure (dbar)")
plt.title("Linear Temperature Trend with MEAN Contours (°C/yr) through Time (Depth vs Latitude)")
plt.gca().invert_yaxis()  # Invert y-axis for depth display
plt.show()
