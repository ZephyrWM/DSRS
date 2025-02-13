import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import pandas as pd
from scipy.stats import linregress

# **定义数据目录**
data_dir = r"D:\DSRS\temp\RG_ArgoClim_Temperature_2019"

# **正则匹配文件名**
pattern_monthly = re.compile(r"RG_ArgoClim_(\d{6})_\d{4}\.nc")  # 解析 201901_2019 这种格式
pattern_annual = re.compile(r"RG_ArgoClim_Temperature_(\d{4})\.nc")  # 解析 Temperature_2019

# **存储月度温度异常**
monthly_anomalies = {}

# **修改纬度范围 (-65°S 到 -40°S)**
lat_range = (-65, -40)

def process_file(file_path, filename):
    global monthly_anomalies

    ds = xr.open_dataset(file_path, decode_times=False)

    # **筛选新的纬度范围**
    ds = ds.sel(LATITUDE=slice(lat_range[0], lat_range[1]))

    # **去掉深度限制，保留所有深度**
    ds_no_depth_limit = ds  # 保留所有深度数据

    if pattern_annual.match(filename):
        print(f"正在处理 {filename}，包含 2004-2018 数据")

        time_values = ds["TIME"].values  # months since 2004-01-01
        start_date = pd.Timestamp("2004-01-01")
        time_decoded = [start_date + pd.DateOffset(months=int(t)) for t in time_values]
        months_decoded = [t.strftime("%Y-%m") for t in time_decoded]
        ds_no_depth_limit = ds_no_depth_limit.assign_coords(TIME=("TIME", months_decoded))

        for month in months_decoded:
            temp_anomaly = ds_no_depth_limit["ARGO_TEMPERATURE_ANOMALY"].sel(TIME=month).mean(dim=["LONGITUDE", "PRESSURE"]).mean().item()
            monthly_anomalies[month] = temp_anomaly

    elif pattern_monthly.match(filename):
        match = pattern_monthly.search(filename)
        if match:
            month_str = match.group(1)
            year = int(month_str[:4])
            month = int(month_str[4:])
            month_key = f"{year}-{month:02d}"
            print(f"正在处理 {filename}，日期: {month_key}")
            temp_anomaly = ds_no_depth_limit["ARGO_TEMPERATURE_ANOMALY"].mean(dim=["LONGITUDE", "PRESSURE"]).mean().item()
            monthly_anomalies[month_key] = temp_anomaly

    ds.close()

# **遍历数据文件**
for filename in tqdm(sorted(os.listdir(data_dir))):
    file_path = os.path.join(data_dir, filename)
    process_file(file_path, filename)

# **确保月份排序**
months = sorted(monthly_anomalies.keys())
global_temp_anomalies = [monthly_anomalies[m] for m in months]

# **创建 DataFrame**
df = pd.DataFrame({"Month": months, "Temp_Anomaly": global_temp_anomalies})
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)

# **计算 12 个月移动平均**
df["Annual_Trend"] = df["Temp_Anomaly"].rolling(window=12, center=True).mean()

# **线性回归分析**
months_numeric = np.arange(len(df))
slope, intercept, r_value, p_value, std_err = linregress(months_numeric, df["Temp_Anomaly"])
df["Linear_Trend"] = intercept + slope * months_numeric

# **输出回归分析结果**
regression_results = pd.DataFrame({
    "Statistic": ["Slope", "Intercept", "R-value", "P-value", "Std Error"],
    "Value": [slope, intercept, r_value, p_value, std_err]
})

# **创建图像**
plt.style.use("seaborn-v0_8-poster")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# **主图：时间序列**
ax1.plot(df.index, df["Temp_Anomaly"], color='black', label="Monthly Temperature Anomaly")
ax1.plot(df.index, df["Annual_Trend"], color='red', linewidth=2, label="12-Month Moving Average")
ax1.plot(df.index, df["Linear_Trend"], color='blue', linestyle='dashed', label="Linear Trend")

ax1.set_title("Southern Ocean Temperature Anomalies (-65°S to -40°S)", fontsize=18, weight='bold')
ax1.set_xlabel("Year", fontsize=14)
ax1.set_ylabel("Temperature Anomaly (°C)", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# **子图：线性回归结果表格**
ax2.axis('tight')
ax2.axis('off')
ax2.table(cellText=regression_results.values,
          colLabels=regression_results.columns,
          cellLoc='center', loc='center', bbox=[0.2, 0.1, 0.6, 0.8])

plt.tight_layout()
plt.show()
