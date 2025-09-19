# pip install chromadb

api_key = "api key를 입력하세요"

from openai import OpenAI
import chromadb
import os
import glob
import pandas as pd

# 1) OpenAI & Chroma 초기화
client = OpenAI(api_key=api_key)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="company_docs")

# === 2) CSV 파일 읽어서 등록 ===
''' csv.파일 구조
content
사내 복지 규정: 연차는 15일에서 시작하며...
IT 보안 정책: 비밀번호는 12자 이상...
재택근무 지침: 매주 최대 2회 재택근무 가능...
'''


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
            if not text:
                continue

            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding

            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[emb],
                documents=[text]
            )
            doc_id += 1

        print(f"[✔] {file} 등록 완료 ({len(df)}행)")
    except Exception as e:
        print(f"[오류] {file}: {e}")

print(f"총 {doc_id}개의 문서를 Chroma에 저장했습니다.\n")

# 3) 검색 + GPT 답변 함수
def ask(query: str) -> str:
    """질문을 받아 문서 검색 후 GPT로 답변 생성"""
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = collection.query(query_embeddings=[q_emb], n_results=3)

    # 여러 문서를 연결
    context = "\n".join(sum(results["documents"], []))
    messages = [
        {"role": "system", "content": "You are a helpful assistant for company policies."},
        {"role": "user", "content": f"아래 문서를 참고해서 질문에 답하세요.\n\n문서:\n{context}\n\n질문: {query}"}
    ]

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    )
    return resp.choices[0].message.content

# 4) 메인 루프 (콘솔 챗봇)
def main():
    print("=== 회사 문서 Q&A 챗봇 ===")
    print("종료하려면 'quit' 또는 'exit' 입력\n")

    while True:
        q = input("질문> ").strip()
        if q.lower() in ("quit", "exit"):
            print("챗봇을 종료합니다.")
            break

        try:
            answer = ask(q)
            print(f"답변> {answer}\n")
        except Exception as e:
            print(f"[오류] {e}\n")

if __name__ == "__main__":
    main()
