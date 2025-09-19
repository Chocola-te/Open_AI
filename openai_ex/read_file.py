import chromadb
import os
import glob
import pandas as pd

DATA_DIR = "data"   # <-- CSV 파일들이 있는 폴더
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

doc_id = 0
for file in csv_files:
    try:
        df = pd.read_csv(file)

        # content 라는 컬럼이 있다고 가정 (없으면 df.columns 확인 후 수정)
        if "content" not in df.columns:
            print(f"{file}: 'content' 컬럼을 찾을 수 없습니다.")
            continue

        for row in df["content"].dropna():
            text = str(row).strip()
            print(text)
            if not text:
                continue
            doc_id += 1

    except Exception as e:
        print(f"[오류] {file}: {e}")
