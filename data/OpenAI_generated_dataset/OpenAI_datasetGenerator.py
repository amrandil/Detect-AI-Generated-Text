# imports
import pandas as pd
from openai import OpenAI
import csv


# output file
csv_file_path = "generated_text.csv"


BASE_PATH = '/Users/amrkandil/Documents/amrkandil/UNI/MSc/ITDS - Study Program/Semesters/Fall2024/ML&DS/practical_work/Project/llm-detect-ai-generated-text'


train_data = pd.read_csv(f'{BASE_PATH}/train_essays.csv')

train_prompts = pd.read_csv(f'{BASE_PATH}/train_prompts.csv')


# Source text and prompts
data = [
    {
        "source_text": row["source_text"],
        "prompt_id": row["prompt_id"],
        "prompt": f"{row['instructions']} (Topic: {row['prompt_name']})"
    }
    for _, row in train_prompts.iterrows()
]


client = OpenAI()


def generate_responses(source_text, prompt, prompt_id, num_responses=700, index=[0]):

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        for i in range(num_responses):
            try:
                # Generate the response
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",
                            "content": f"Source Text: {source_text}\nPrompt: {prompt}"}
                    ],
                    temperature=0.8,  # Encourage creative and varied responses
                    max_tokens=800    # Ensure output length is approximately 500-600 words
                )
                generated_text = response.choices[0].message.content

                writer.writerow([index[0], prompt_id, generated_text, 1])

                print(
                    f"Generated response {index[0]}, generated: {generated_text:.20}")
            except Exception as e:
                print(f"Error on iteration {i}: {e}")

                writer.writerow(
                    [index[0], prompt_id, "Error generating response", 1])
            index[0] += 1
    return


# Generate responses for each source text and prompt
index = [0]
for item in data:
    print(f"Generating responses for prompt: {item['prompt_id']}")
    responses = generate_responses(
        item['source_text'], item['prompt'], item['prompt_id'], num_responses=700, index=index)
