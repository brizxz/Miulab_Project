import json
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Load questions from JSON file
with open("law_qa.json", "r") as file:
    data = json.load(file)

questions = data["questions"]

# Function to get the model's answer
def get_model_answer(question):
    prompt = f"""
    問題: {question}
    
    請以中華民國（台灣）的法律簡答正確答案。
    """
    completion = client.chat.completions.create(
        model="yentinglin/llama-3-taiwan-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    model_answer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            model_answer += chunk.choices[0].delta.content

    return model_answer.strip()

model_answers = []
for idx, q in enumerate(questions):
    question = q["question"]
    print(f"Processing Question {idx + 1}...")
    model_answer = get_model_answer(question)
    model_answers.append({
        "question": question,
        "model_answer": model_answer,
        "correct_answer": q.get("official_answer", "N/A")  
    })

# Save the results to a file
with open("law_qa_llamataiwan.json", "w", encoding="utf-8") as output_file:
    json.dump(model_answers, output_file, ensure_ascii=False, indent=4)

print("Model answers collected and saved")