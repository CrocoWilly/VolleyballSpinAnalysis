import os
import json
import pandas as pd

# 設定 JSON 資料夾和輸出 Excel 檔案
json_folder = "json_results"
output_excel = "1015-01_30_ball_data.xlsx"

# 創建一個空的 DataFrame
df_all = pd.DataFrame(columns=["source", "frame_id", "state", "vel_kmh", "spin_rpm"])

# 記錄 frame_id 的累計偏移量
frame_offset = 0

# 遍歷 json_results 資料夾中的所有 JSON 檔案
for idx, filename in enumerate(sorted(os.listdir(json_folder))):  # 確保按順序處理
    if filename.endswith(".json"):
        file_path = os.path.join(json_folder, filename)
        print(f"處理檔案: {file_path}")

        # 讀取 JSON 檔案
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)  # 解析 JSON
                
                # 確保 JSON 內有 "ball_data" 並且是列表
                if "ball_data" in data and isinstance(data["ball_data"], list):
                    df = pd.DataFrame(data["ball_data"])  # 轉換為 DataFrame
                    
                    # 選擇所需欄位
                    df = df[["frame_id", "state", "vel_kmh", "spin_rpm"]].copy()
                    
                    # **處理 `vel_kmh`，確保為整數，`NaN` 設為 0**
                    df["vel_kmh"] = df["vel_kmh"].fillna(0).astype(int)

                    # **新增 `source` 欄位作為第一欄**
                    df.insert(0, "source", filename)  # 插入到 DataFrame 第一欄
                    
                    # **確保 `frame_id` 唯一**
                    df["frame_id"] += frame_offset  # 為 frame_id 加上偏移量
                    
                    # 更新 frame_offset 為該批次最大 frame_id + 1
                    frame_offset = df["frame_id"].max() + 1

                    # 合併到總表
                    df_all = pd.concat([df_all, df], ignore_index=True)
                else:
                    print(f"⚠️ 警告: {filename} 缺少 'ball_data' 或格式不正確。")
            except json.JSONDecodeError:
                print(f"❌ 錯誤: {filename} 解析失敗，可能不是有效的 JSON 格式。")

# 存成 Excel
df_all.to_excel(output_excel, index=False, engine="openpyxl")
print(f"✅ 所有資料已整合至 {output_excel}")
