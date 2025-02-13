import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import pandas as pd
from scipy.stats import linregress

# **定义数据目录**
data_dir = r"D:\DSRS\temp\RG_ArgoClim_Salinity_2019"

# **正则匹配文件名**
pattern_monthly = re.compile(r"RG_ArgoClim_(\d{6})_\d{4}\.nc")  # 匹配 201901_2019 这样的文件
pattern_annual = re.compile(r"RG_ArgoClim_Salinity_(\d{4})\.nc")  # 匹配 Salinity_2019

# **存储月度盐度异常**
monthly_anomalies = {}

# **修改纬度范围 (-65°S 到 -40°S)**
lat_range = (-65, -40)

def process_file(file_path, filename):
    global monthly_anomalies
    
    ds = xr.open_dataset(file_path, decode_times=False)
    
    # **筛选新的纬度范围**
    ds = ds.sel(LATITUDE=slice(lat_range[0], lat_range[1]))
    
    if pattern_annual.match(filename):
        print(f"正在处理 {filename}，包含 2004-2018 数据")
        
        time_values = ds["TIME"].values  # months since 2004-01-01
        start_date = pd.Timestamp("2004-01-01")
        time_decoded = [start_date + pd.DateOffset(months=int(t)) for t in time_values]
        months_decoded = [t.strftime("%Y-%m") for t in time_decoded]

        ds = ds.assign_coords(TIME=("TIME", months_decoded))

        for month in months_decoded:
            # **去掉深度限制，对深度维度进行平均**
            sal_anomaly = ds["ARGO_SALINITY_ANOMALY"].sel(TIME=month).mean(dim=["LONGITUDE", "PRESSURE"]).mean().item()
            monthly_anomalies[month] = sal_anomaly
    
    elif pattern_monthly.match(filename):
        match = pattern_monthly.search(filename)
        if match:
            month_str = match.group(1)
            year = int(month_str[:4])
            month = int(month_str[4:])
            month_key = f"{year}-{month:02d}"

            print(f"正在处理 {filename}，日期: {month_key}")
            
            # **去掉深度限制，对深度维度进行平均**
            sal_anomaly = ds["ARGO_SALINITY_ANOMALY"].mean(dim=["LONGITUDE", "PRESSURE"]).mean().item()
            monthly_anomalies[month_key] = sal_anomaly
    
    ds.close()


# **遍历数据文件**
for filename in tqdm(sorted(os.listdir(data_dir))):
    file_path = os.path.join(data_dir, filename)
    process_file(file_path, filename)

# **确保月份排序**
months = sorted(monthly_anomalies.keys())  # 确保时间顺序
global_sal_anomalies = [monthly_anomalies[m] for m in months]

# **创建 DataFrame，方便处理数据**
df = pd.DataFrame({"Month": months, "Sal_Anomaly": global_sal_anomalies})
df["Month"] = pd.to_datetime(df["Month"])  # 转换为时间格式
df.set_index("Month", inplace=True)

# **计算 12 个月移动平均**
df["Annual_Trend"] = df["Sal_Anomaly"].rolling(window=12, center=True).mean()

# **线性回归分析**
months_numeric = np.arange(len(df))  # 数字索引时间
slope, intercept, r_value, p_value, std_err = linregress(months_numeric, df["Sal_Anomaly"])  
df["Linear_Trend"] = intercept + slope * months_numeric

# **输出回归分析结果**
print(f"线性回归分析结果：")
print(f"斜率 (Slope): {slope}")
print(f"截距 (Intercept): {intercept}")
print(f"相关系数 (R-value): {r_value}")
print(f"P值 (P-value): {p_value}")
print(f"标准误差 (Std Error): {std_err}")

# **创建图像**
plt.figure(figsize=(12, 5))

# **绘制月度盐度异常（黑色）**
plt.plot(df.index, df["Sal_Anomaly"], color='black', label="Monthly Salinity Anomaly")

# **绘制 12 个月移动平均（红色）**
plt.plot(df.index, df["Annual_Trend"], color='red', linewidth=2, label="12-Month Moving Average")

# **绘制线性趋势（蓝色）**
plt.plot(df.index, df["Linear_Trend"], color='blue', linestyle='dashed', label="Linear Trend")

# **添加标题和图例**
plt.title("Southern Ocean Salinity Anomalies (-65°S to -40°S)")
plt.xlabel("Year")
plt.ylabel("Salinity Anomaly (PSU)")
plt.legend()
plt.grid()

# **显示图像**
plt.show()
