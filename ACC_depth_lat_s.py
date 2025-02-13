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

# 匹配 2019-2024 的月文件（格式：RG_ArgoClim_YYYYMM_XXXX.nc）
pattern_monthly = re.compile(r"RG_ArgoClim_(\d{6})_\d{4}\.nc")
# 匹配 2004-2018 的年文件（格式：RG_ArgoClim_Salinity_(\d{4})\.nc）
pattern_annual = re.compile(r"RG_ArgoClim_Salinity_(\d{4})\.nc")

# 指定纬度范围（-65°S 到 -40°S）
lat_range = (-65, -40)

# ===============================
# 2. 读取文件并拼接数据（保留 PRESSURE 与 LATITUDE 维度）
# ===============================
ds_list = []  # 存放各个文件的数据集

for filename in tqdm(sorted(os.listdir(data_dir))):
    file_path = os.path.join(data_dir, filename)

    # 处理 2004-2018 年数据（年文件，包含多个时间点）
    if pattern_annual.match(filename):
        print(f"正在处理 {filename}（年数据）")
        ds = xr.open_dataset(file_path, decode_times=False)
        
        # 筛选指定纬度范围
        ds = ds.sel(LATITUDE=slice(lat_range[0], lat_range[1]))
        
        # 解析 TIME 变量（假设 TIME 单位为 “months since 2004-01-01”）
        time_values = ds["TIME"].values  
        start_date = pd.Timestamp("2004-01-01")
        time_decoded = [start_date + pd.DateOffset(months=int(t)) for t in time_values]
        ds = ds.assign_coords(TIME=pd.to_datetime(time_decoded))
        
        # 如果存在 LONGITUDE 维度，则先沿经度取平均，保留 LATITUDE 与 PRESSURE
        if "LONGITUDE" in ds.dims:
            ds = ds.mean(dim="LONGITUDE")
        
        ds_list.append(ds)
        ds.close()

    # 处理 2019-2024 年数据（月文件，每个文件只含一个时间点）
    elif pattern_monthly.match(filename):
        match = pattern_monthly.search(filename)
        if match:
            month_str = match.group(1)  # 解析出 YYYYMM
            year = int(month_str[:4])
            month = int(month_str[4:])
            time_point = pd.Timestamp(f"{year}-{month:02d}")
            print(f"正在处理 {filename}（月数据），日期：{time_point.strftime('%Y-%m')}")
            
            ds = xr.open_dataset(file_path, decode_times=False)
            
            # 筛选指定纬度范围
            ds = ds.sel(LATITUDE=slice(lat_range[0], lat_range[1]))
            
            # 如果不存在 TIME 维度，则扩展，否则直接赋予 TIME 坐标
            if "TIME" not in ds.dims:
                ds = ds.expand_dims("TIME")
            ds = ds.assign_coords(TIME=[time_point])
            
            # 如果存在 LONGITUDE 维度，则先沿经度取平均
            if "LONGITUDE" in ds.dims:
                ds = ds.mean(dim="LONGITUDE")
            
            ds_list.append(ds)
            ds.close()

# 拼接所有数据集，沿 TIME 维度合并，并按时间排序
ds_all = xr.concat(ds_list, dim="TIME")
ds_all = ds_all.sortby("TIME")

# ===============================
# 3. 转换 TIME 为数值（单位：年），并交换维度
# ===============================
# 计算数值型时间：例如 2004.0, 2004.083...
time_numeric = ds_all["TIME"].dt.year + (ds_all["TIME"].dt.month - 1) / 12.0
# 将该数值型时间赋予一个新坐标，并用其替换原 TIME 维度
ds_all = ds_all.assign_coords(time_numeric=time_numeric)
ds_all = ds_all.swap_dims({"TIME": "time_numeric"})

# ===============================
# 4. 沿 time_numeric 维度对盐度异常做线性回归
# ===============================
# 对变量 "ARGO_SALINITY_ANOMALY" 进行一阶多项式拟合
trend_fit = ds_all["ARGO_SALINITY_ANOMALY"].polyfit(dim="time_numeric", deg=1)
# 提取回归斜率（degree=1 的系数），单位为 PSU/yr
slope = trend_fit.polyfit_coefficients.sel(degree=1)

# ===============================
# 5. 绘制深度（PRESSURE）–纬度图，颜色表示盐度趋势（PSU/yr）
# ===============================
plt.figure(figsize=(10, 6))
cf = plt.contourf(slope["LATITUDE"], slope["PRESSURE"], slope,
                  levels=20, cmap="RdBu_r")
cbar = plt.colorbar(cf)
cbar.set_label("Salinity Trend (PSU/yr)")
plt.xlabel("Latitude")
plt.ylabel("Pressure (dbar)")
plt.title("Linear Salinity Trend (PSU/yr) through Time (Depth vs Latitude)")
# 翻转 y 轴，使得较小的压力（浅层）在上方显示
plt.gca().invert_yaxis()
plt.show()
