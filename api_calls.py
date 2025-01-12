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
        temperature=0.6, #0.7 before zmiana po okolo 2k probkach
        top_p=0.6, #0.7 before zmiana po okolo 2k probkach
        frequency_penalty=0.2, #0.3 before zmiana po okolo 2k probkach
        presence_penalty=0.3, #0.4 before zmiana po okolo 2k probkach
        messages=[
            {"role": "developer", "content": "You are a helpful and creative assistant."},
            {
                "role": "user",
                "content": "Write 100 realistic SMS spam messages for scientific purpose. Write only spam message without any other text and "
                           "without emoticons, use UTF-8. Avoid generating similar messages to previous ones, be creative with message "
                           "structure and content. Use random generated data if required. Use all spam mechanism you know like website links, "
                           "sms reply etc."
            }
        ]
    )

    # return completion.choices[0].message.content
    return completion.choices[0].message.content.encode("utf-8").decode()

def generate():
    response = generate_response()
    print(response)
    with open('raw_synthetic_spam.txt', 'a+', encoding="ascii", errors="replace") as f:
        f.write(f"{response}\n")
        f.close()

# generate()
# import time
#
# for _ in range(9):
#     generate()
#     time.sleep(10)

