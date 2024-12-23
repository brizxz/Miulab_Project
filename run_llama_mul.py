import json
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Load questions from JSON file
with open("law_mul.json", "r") as file:
    data = json.load(file)

questions = data["questions"]

# Function to get the model's answer
def get_model_answer(question, options):
    prompt = f"""
    問題: {question}
    選項:
    {chr(65)}. {options[0]}
    {chr(66)}. {options[1]}
    {chr(67)}. {options[2]}
    {chr(68)}. {options[3]}
    
    請以選項(A, B, C, D)回答正確答案。
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

def extract_answer_letter(model_output):
    for char in model_output.strip().upper():
        if char in {"A", "B", "C", "D"}:
            return char
    return None

results = []
# Process each question
for idx, q in enumerate(questions):
    question = q["question"]
    options = q["options"]
    correct_answer_letter = q["answer"].strip().upper()

    model_output = get_model_answer(question, options)
    model_answer = extract_answer_letter(model_output)


    # Compare and print the result
    is_correct = model_answer == correct_answer_letter
    result = {
        "question": question,
        "options": options,
        "model_answer": model_answer,
        "correct_answer": correct_answer_letter,
        "is_correct": is_correct
    }
    results.append(result)


# Save all results to a JSON file
with open("law_mul_llamataiwan.json", "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
