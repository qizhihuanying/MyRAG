import csv
import json
import random
import re
from ollama import Client
from tqdm import tqdm
from lightrag import LightRAG
from lightrag.prompt import PROMPTS

def ollama_complete_if_cache(
    model="qwen2.5:7b", prompt=None, system_prompt=None, history_messages=[], **kwargs
) -> str:
    host = kwargs.pop("host", "http://localhost:11434") 
    ollama_client = Client(host=host)  
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = ollama_client.chat(model=model, messages=messages, **kwargs)
    return response["message"]["content"]

def load_answers(cls, mode):
    file_path = f'./query_results/{cls}/result_{mode}.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_answers(data):
    answers = []
    for item in data:
        answers.append((item['query'], item['result']))
    return answers

def evaluate_answers(query, answer_1, answer_2):
    prompt = PROMPTS["evaluate"].format(answer_1=answer_1, answer_2=answer_2, query=query)
    response = ollama_complete_if_cache(prompt=prompt)
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            evaluation_json = json_match.group(0)
            evaluation_result = json.loads(evaluation_json)
        else:
            print("Failed to find JSON formatted evaluation result in the response.")
            evaluation_result = None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        evaluation_result = None
    return evaluation_result

def compare_modes(cls):
    hybrid_data = load_answers(cls, 'hybrid')
    naive_data = load_answers(cls, 'naive')
    hybrid_answers = extract_answers(hybrid_data)
    naive_answers = extract_answers(naive_data)
    hybrid_wins = {'comprehensiveness': 0, 'diversity': 0, 'empowerment': 0, 'overall': 0}
    naive_wins = {'comprehensiveness': 0, 'diversity': 0, 'empowerment': 0, 'overall': 0}
    for (query_hybrid, answer_hybrid), (query_naive, answer_naive) in tqdm(zip(hybrid_answers, naive_answers), desc=f"Evaluating {cls}", total=len(hybrid_answers)):
        if query_hybrid != query_naive:
            print(f"Warning: Mismatched questions detected! ({query_hybrid} != {query_naive})")
            continue
        if random.choice([True, False]):
            answer1, answer2 = answer_hybrid, answer_naive
            answer1_label, answer2_label = 'hybrid', 'naive'
        else:
            answer1, answer2 = answer_naive, answer_hybrid
            answer1_label, answer2_label = 'naive', 'hybrid'
        evaluation_result = evaluate_answers(query_hybrid, answer1, answer2)
        if evaluation_result is None:
            continue
        for criterion in ['comprehensiveness', 'diversity', 'empowerment', 'overall']:
            winner = evaluation_result.get(criterion, '').lower()
            if winner == 'answer 1':
                if answer1_label == 'hybrid':
                    hybrid_wins[criterion] += 1
                else:
                    naive_wins[criterion] += 1
            elif winner == 'answer 2':
                if answer2_label == 'hybrid':
                    hybrid_wins[criterion] += 1
                else:
                    naive_wins[criterion] += 1
            else:
                print(f"Unrecognized evaluation result: {winner} (Dimension: {criterion})")
    total_questions = len(hybrid_answers)
    results = []
    for criterion in ['comprehensiveness', 'diversity', 'empowerment', 'overall']:
        naive_rate = 100 * naive_wins[criterion] / total_questions
        hybrid_rate = 100 * hybrid_wins[criterion] / total_questions
        results.append([cls, criterion, f"{naive_rate:.2f}%", f"{hybrid_rate:.2f}%"])
    return results

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'Dimension', 'NaiveRAG', 'LightRAG'])
        for result in results:
            writer.writerow(result)

def main():
    categories = ['agriculture', 'cs', 'legal', 'mix']
    all_results = []
    for category in categories:
        category_results = compare_modes(category)
        all_results.extend(category_results)
    save_results_to_csv(all_results, './reproduce/evaluation_results.csv')

main()