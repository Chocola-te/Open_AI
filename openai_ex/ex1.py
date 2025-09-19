from openai import OpenAI
api_key = "발급한 api key"

client = OpenAI(api_key=api_key)


response = client.chat.completions.create(
    model="gpt-3.5-turbo", # 모델의 성격을 지정 (예: gpt-3.5-turbo, gpt-4 등)
    messages=[ # 사용자의 입력 메시지
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "신기하네"}
    ]

)

print(response.choices[0].message.content)