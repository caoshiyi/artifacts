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
    for i in range(12):
        for question in questions:
                prompts.append(question["turns"][0])


    warmup = requests.post(
        url + "/generate",
        json={
            "text": prompts[:50],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
            },
            "batch": True,
        },
    )


    start = time.time()
    response = requests.post(
        url + "/generate",
        json={
            "text": prompts[:960],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
            },
            "batch": True,
        },
    )
    end = time.time()
    # save the result, create the file if it doesn't exist
    print(response.json())
    print(f"Time: {end - start:.3f}s")
    print(f"Throughput: {len(prompts) / (end - start):.3f} prompts/s")
    with open("response.json", "w") as fout:
        json.dump(response.json(), fout)
