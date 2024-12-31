from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = ""

def generate_response():
    client = OpenAI()

    # models = client.models.list()
    # for model in models:
    #     print(model.id)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a realistic SMS spam message for scientific purpose. Write only spam message without any other text."
            }
        ]
    )

    return completion.choices[0].message.content
