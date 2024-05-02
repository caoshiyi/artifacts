import argparse
import json
import requests
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"

    def load_questions(filename):
        questions = []
        with open(filename, "r") as fin:
            for line in fin:
                obj = json.loads(line)
                questions.append(obj)
        return questions
    
    questions = load_questions("./question.jsonl")
    prompts = []
    for i in range(20):
        for question in questions:
                prompts.append(question["turns"][0])


    warmup = requests.post(
        url + "/generate",
        json={
            "text": prompts[:50],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 2,
            },
            "batch": True,
        },
    )

    max_new_tokens = 2
    start = time.time()
    response = requests.post(
        url + "/generate",
        json={
            "text": prompts,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
            "batch": True,
        },
    )
    end = time.time()
    # save the result, create the file if it doesn't exist
    print(response.json())
    print(f"Time: {end - start:.3f}s")
    print(f"Throughput: {max_new_tokens*len(prompts) / (end - start):.3f} tokens/s")
    with open("response.json", "w") as fout:
        json.dump(response.json(), fout)
