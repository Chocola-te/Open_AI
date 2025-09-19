# pip install chromadb

api_key = "API키를 입력하세요"


from openai import OpenAI
import chromadb
import os

# 1) OpenAI & Chroma 초기화
client = OpenAI(api_key=api_key)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="company_docs")

# 2) 사내 문서 등록 (예제)
docs = [
    "사내 복지 규정: 연차는 15일에서 시작하며, 근속 연수에 따라 증가합니다.",
    "IT 보안 정책: 비밀번호는 12자 이상이며, 3개월마다 변경해야 합니다.",
    "재택근무 지침: 매주 최대 2회 재택근무가 가능합니다.",
]

# 문서 → 임베딩 → DB에 추가
for i, d in enumerate(docs):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=d
    ).data[0].embedding
    collection.add(ids=[f"doc_{i}"], embeddings=[emb], documents=[d])

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
