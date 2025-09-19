from openai import OpenAI
import os

api_key = "발급한 api key"

# API 키 설정 (환경 변수 사용 권장)
client = OpenAI(api_key=api_key)

def chat_with_gpt(messages):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.7  # 창의성 조절
  )
  return response.choices[0].message.content

def main():
  print("=== 미니 챗봇 시작 ===")
  messages = [{"role": "system", "content": "You are a helpful assistant."}]

  while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
      print("챗봇 종료!")
      break

    messages.append({"role": "user", "content": user_input})
    bot_reply = chat_with_gpt(messages)
    messages.append({"role": "assistant", "content": bot_reply})

    print("Bot:", bot_reply)

if __name__ == "__main__":
  main()