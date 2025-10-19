import json
import pandas as pd

dir_path = "./test_results/HDR80_A_Live_20231015_182253_000.mov_detect/"

# 讀取 JSON 檔案
file_path = dir_path + "ball_data.json"  # 請替換為你的 JSON 檔案路徑

with open(file_path, "r") as file:
    data = json.load(file)

# 提取所需的數據
frames = []
for frame in data.get("ball_data", []):
    frame_id = frame.get("frame_id")
    state = frame.get("state")
    vel_kmh = frame.get("vel_kmh")
    spin_rpm = frame.get("spin_rpm")

    vel_kmh = round(vel_kmh) if isinstance(vel_kmh, (int, float)) else None

    frames.append({"frame_id": frame_id, "state": state, "vel_kmh": vel_kmh, "spin_rpm": spin_rpm})

# 轉換為 DataFrame
df = pd.DataFrame(frames)

# 將結果存為 CSV
csv_path = dir_path + "ball_data.csv"  # 輸出的 CSV 檔案名稱
df.to_csv(csv_path, index=False)
print(f"CSV 檔案已儲存至: {csv_path}")


# 將結果存為 Excel
xlsx_path = dir_path + "ball_data.xlsx"  # 輸出的 Excel 檔案名稱
df.to_excel(xlsx_path, index=False, engine="openpyxl")
print(f"Excel 檔案已儲存至: {xlsx_path}")
