import csv
import json
import os
import random
import re
from ollama import Client
from tqdm import tqdm
from lightrag import LightRAG
from lightrag.prompt import PROMPTS

def ollama_complete_if_cache(model="qwen2.5:7b", prompt=None, system_prompt=None, history_messages=[], **kwargs) -> str:
    host = kwargs.pop("host", "http://localhost:11434")
    ollama_client = Client(host=host)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = ollama_client.chat(model=model, messages=messages, **kwargs)
    return response["message"]["content"]

def load_answers(cls, model):
    file_path = f'./results/{model}/query_results/{cls}/result.json'
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

def compare_modes(cls, num_runs=5):
    multilight_data = load_answers(cls, 'multi_lightrag')
    light_data = load_answers(cls, 'lightrag')
    multilight_answers = extract_answers(multilight_data)
    light_answers = extract_answers(light_data)
    total_questions = len(multilight_answers)

    multilight_wins_cumulative = {'comprehensiveness': 0, 'diversity': 0, 'empowerment': 0, 'overall': 0}
    light_wins_cumulative = {'comprehensiveness': 0, 'diversity': 0, 'empowerment': 0, 'overall': 0}

    for run in range(num_runs):
        print(f"Run {run+1} of {num_runs}")
        multilight_wins = {'comprehensiveness': 0, 'diversity': 0, 'empowerment': 0, 'overall': 0}
        light_wins = {'comprehensiveness': 0, 'diversity': 0, 'empowerment': 0, 'overall': 0}
        for (query_multilight, answer_multilight), (query_light, answer_light) in tqdm(zip(multilight_answers, light_answers), desc=f"Evaluating {cls} Run {run+1}", total=len(multilight_answers)):
            if query_multilight != query_light:
                print(f"Warning: Mismatched questions detected! ({query_multilight} != {query_light})")
                continue
            if random.choice([True, False]):
                answer1, answer2 = answer_multilight, answer_light
                answer1_label, answer2_label = 'multilight', 'light'
            else:
                answer1, answer2 = answer_light, answer_multilight
                answer1_label, answer2_label = 'light', 'multilight'
            evaluation_result = evaluate_answers(query_multilight, answer1, answer2)
            if evaluation_result is None:
                continue
            for criterion in ['comprehensiveness', 'diversity', 'empowerment', 'overall']:
                winner = evaluation_result.get(criterion, '').lower()
                if winner == 'answer 1':
                    if answer1_label == 'multilight':
                        multilight_wins[criterion] += 1
                    else:
                        light_wins[criterion] += 1
                elif winner == 'answer 2':
                    if answer2_label == 'multilight':
                        multilight_wins[criterion] += 1
                    else:
                        light_wins[criterion] += 1
                else:
                    print(f"Unrecognized evaluation result: {winner} (Dimension: {criterion})")
        for criterion in ['comprehensiveness', 'diversity', 'empowerment', 'overall']:
            multilight_wins_cumulative[criterion] += multilight_wins[criterion]
            light_wins_cumulative[criterion] += light_wins[criterion]

    results = []
    for criterion in ['comprehensiveness', 'diversity', 'empowerment', 'overall']:
        light_avg_wins = light_wins_cumulative[criterion] / num_runs
        multilight_avg_wins = multilight_wins_cumulative[criterion] / num_runs
        light_rate = 100 * light_avg_wins / total_questions
        multilight_rate = 100 * multilight_avg_wins / total_questions
        results.append([cls, criterion, f"{light_rate:.2f}%", f"{multilight_rate:.2f}%"])
    return results

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'Dimension', 'LightRAG', 'Multi-LightRAG'])
        for result in results:
            writer.writerow(result)

def main():
    categories = ['agriculture', 'cs', 'legal', 'mix']
    all_results = []
    for category in categories:
        category_results = compare_modes(category, num_runs=3)
        all_results.extend(category_results)
    os.makedirs('./results/result_csvs', exist_ok=True)
    save_results_to_csv(all_results, './results/result_csvs/light_vs_multilight_results.csv')

main()
