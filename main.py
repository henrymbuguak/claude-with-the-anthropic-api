from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # Load environment variables from .env file

client = Anthropic()
model = "claude-sonnet-4-6"  # Specify the model you want to use'

# multi-turn conversation example


def add_user_message(messages, user_input):
    messages.append({
        "role": "user",
        "content": user_input
    })


def add_assistant_message(messages, assistant_response):
    messages.append({
        "role": "assistant",
        "content": assistant_response
    })


def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens=500,
        messages=messages
    )
    return message.content[0].text


messages = []

add_user_message(
    messages, "What's applied AI and how do I become an applied AI engineer?")
answer = chat(messages)
add_assistant_message(messages, answer)
add_user_message(messages, "What are the best resources to learn applied AI?")
answer = chat(messages)
print(answer)
