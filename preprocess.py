"""
AI Hub 낚시성 기사 탐지 데이터 전처리 스크립트
실행: python preprocess.py
결과: clickbait_data.csv (title, category, clickbait_class)
"""

import zipfile
import json
import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\jiwoo-choi\Downloads\146.낚시성 기사 탐지 데이터\01-1.정식개방데이터")
OUTPUT_PATH = Path(r"C:\Users\jiwoo-choi\Desktop\clickbait_detection\clickbait_data.csv")

LABEL_DIRS = [
    BASE_DIR / "Training"   / "02.라벨링데이터",
    BASE_DIR / "Validation" / "02.라벨링데이터",
]

def parse_json(data: dict, source: str) -> dict | None:
    try:
        src   = data["sourceDataInfo"]
        label = data["labeledDataInfo"]
        return {
            "title":           src["newsTitle"],
            "category":        src.get("newsCategory", ""),
            "clickbait_class": int(label["clickbaitClass"]),
            "split":           source,
        }
    except (KeyError, TypeError):
        return None

records = []
for label_dir in LABEL_DIRS:
    split = "train" if "Training" in str(label_dir) else "val"
    zip_files = list(label_dir.glob("*.zip"))
    print(f"\n[{split}] {len(zip_files)}개 zip 처리 중...")

    for i, zip_path in enumerate(zip_files, 1):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if not name.endswith(".json"):
                        continue
                    with zf.open(name) as f:
                        data = json.load(f)
                        row = parse_json(data, split)
                        if row:
                            records.append(row)
        except Exception as e:
            print(f"  오류 ({zip_path.name}): {e}")

        if i % 10 == 0:
            print(f"  {i}/{len(zip_files)} 완료 | 누적: {len(records):,}건")

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n완료! 총 {len(df):,}건 저장 → {OUTPUT_PATH}")
print("\n클래스 분포:")
print(df["clickbait_class"].value_counts())
print("\n분할 분포:")
print(df["split"].value_counts())
