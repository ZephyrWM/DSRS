import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import pandas as pd

# ===============================
# 1. 定义数据目录和文件名正则表达式
# ===============================
data_dir = r"D:\DSRS\temp\RG_ArgoClim_Salinity_2019"

# 匹配 2019-2024 的月文件
pattern_monthly = re.compile(r"RG_ArgoClim_(\d{6})_\d{4}\.nc")
# 匹配 2004-2018 的年文件
pattern_annual = re.compile(r"RG_ArgoClim_Salinity_(\d{4})\.nc")

# 纬度范围（南纬 40°-65°，每 5° 取一个点）
latitudes = np.arange(-65, -39, 5)

# ===============================
# 2. 读取文件并拼接数据（保留 PRESSURE、LONGITUDE 和 LATITUDE 维度）
# ===============================
ds_list = []  # 存放各个文件的数据集

for filename in tqdm(sorted(os.listdir(data_dir))):
    file_path = os.path.join(data_dir, filename)

    if pattern_annual.match(filename) or pattern_monthly.match(filename):
        print(f"正在处理 {filename}")
        ds = xr.open_dataset(file_path, decode_times=False)
        
        if pattern_annual.match(filename):
            time_values = ds["TIME"].values  
            start_date = pd.Timestamp("2004-01-01")
            time_decoded = [start_date + pd.DateOffset(months=int(t)) for t in time_values]
            ds = ds.assign_coords(TIME=pd.to_datetime(time_decoded))
        elif pattern_monthly.match(filename):
            match = pattern_monthly.search(filename)
            month_str = match.group(1)  
            year = int(month_str[:4])
            month = int(month_str[4:])
            time_point = pd.Timestamp(f"{year}-{month:02d}")
            ds = ds.assign_coords(TIME=[time_point])
        
        ds = ds.sel(LATITUDE=slice(-65, -40))  # 先选定纬度范围
        ds_list.append(ds)
        ds.close()

# 拼接所有数据集
ds_all = xr.concat(ds_list, dim="TIME")
ds_all = ds_all.sortby("TIME")

# ===============================
# 3. 逐个纬度绘制深度（PRESSURE）–经度图
# ===============================
for lat in latitudes:
    plt.figure(figsize=(10, 6))
    ds_lat = ds_all.sel(LATITUDE=lat, method="nearest").mean(dim="TIME")
    cf = plt.contourf(ds_lat["LONGITUDE"], ds_lat["PRESSURE"], ds_lat["ARGO_SALINITY_ANOMALY"], levels=20, cmap="RdBu_r")
    cbar = plt.colorbar(cf)
    cbar.set_label("Salinity Anomaly (PSU)")
    plt.xlabel("Longitude")
    plt.ylabel("Pressure (dbar)")
    plt.title(f"Depth vs Longitude Salinity Anomaly at {lat}°S")
    plt.gca().invert_yaxis()
    plt.show()
