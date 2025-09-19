from openai import OpenAI
import chromadb

api_key = "발급한 api key"

client = OpenAI(api_key=api_key)

# 1) Chroma DB 초기화
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="company_docs")

# 2) 문서 임베딩 후 DB에 추가
docs = [
    "사내 복지 규정: 연차는 5일에서 시작…",
    "IT 보안 정책: 비밀번호는 12자 이상…",
]

for i, d in enumerate(docs):
    emb = client.embeddings.create(model="text-embedding-3-small", input=d).data[0].embedding
    collection.add(ids=[f"doc_{i}"], embeddings=[emb], documents=[d])

# 3) 검색 + GPT 답변
def ask(query):
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=2)

    context = "\n".join(results["documents"][0])
    messages = [
        {"role": "system", "content": "You are a helpful assistant for company policies."},
        {"role": "user", "content": f"문서 내용:\n{context}\n\n질문: {query}"}
    ]
    resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return resp.choices[0].message.content

print(ask("연차는 며칠부터 시작해?"))
