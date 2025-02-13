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
data_dir = r"D:\DSRS\temp\RG_ArgoClim_Temperature_2019"

pattern_monthly = re.compile(r"RG_ArgoClim_(\d{6})_\d{4}\.nc")
pattern_annual = re.compile(r"RG_ArgoClim_Temperature_(\d{4})\.nc")

latitudes = np.arange(-65, -39, 5)  # 纬度范围（南纬 40°-65°）

# ===============================
# 2. 读取文件并拼接数据
# ===============================
ds_list = []

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
        
        ds = ds.sel(LATITUDE=slice(-65, -40))  # 选定纬度范围
        ds_list.append(ds)
        ds.close()

# 拼接数据集并排序
ds_all = xr.concat(ds_list, dim="TIME")
ds_all = ds_all.sortby("TIME")

# 计算平均温度
mean_temperature = ds_all["ARGO_TEMPERATURE_MEAN"].mean(dim="TIME")

# ===============================
# 3. 逐个纬度绘制深度（PRESSURE）–经度图，添加平均温度等势线，居中颜色条
# ===============================
for lat in latitudes:
    plt.figure(figsize=(12, 8))
    ds_lat = ds_all.sel(LATITUDE=lat, method="nearest").mean(dim="TIME")
    
    # 设置颜色条范围，确保 0 为中心
    vmin, vmax = -0.18, 0.18  # 根据温度异常实际范围调整
    cf = plt.contourf(ds_lat["LONGITUDE"], ds_lat["PRESSURE"], ds_lat["ARGO_TEMPERATURE_ANOMALY"],
                      levels=np.linspace(vmin, vmax, 21), cmap="RdBu_r", extend='both')
    cbar = plt.colorbar(cf)
    cbar.set_label("Temperature Anomaly (°C)")

    # 叠加平均温度等势线
    contour_levels = np.arange(-2, 30, 2)  # 温度等势线范围和间隔
    mean_temperature_lat = mean_temperature.sel(LATITUDE=lat, method="nearest")
    mean_contours = plt.contour(mean_temperature_lat["LONGITUDE"], mean_temperature_lat["PRESSURE"],
                                mean_temperature_lat, levels=contour_levels, colors='black', linewidths=0.75, alpha=0.7)
    plt.clabel(mean_contours, inline=True, fontsize=8, fmt="%.1f")

    # 设置图表标题和标签
    plt.xlabel("Longitude")
    plt.ylabel("Pressure (dbar)")
    plt.title(f"Depth vs Longitude Temperature Anomaly with Mean Temperature Contours at {lat}°S")
    plt.gca().invert_yaxis()
    plt.show()
