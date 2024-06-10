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
    for i in range(59):
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
            "max_padding_length": 0,
        },
    )
    print(warmup.json())

    max_new_tokens = 32
    max_padding_length = 0
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
            "max_padding_length": max_padding_length,
        },
    )
    end = time.time()
    # save the result, create the file if it doesn't exist
    print(response.json())
    num_finished = 0
    for req in response.json():
         if req['meta_info']['completion_tokens'] == max_new_tokens:
            num_finished += 1
        
    print(f"Time: {end - start:.3f}s")
    print(f"Throughput: {max_new_tokens*num_finished / (end - start):.3f} tokens/s")
    results = {"time": end - start, "throughput": max_new_tokens*num_finished / (end - start), "max_new_tokens": max_new_tokens, "max_padding_length": max_padding_length}
    # write the results to a file, create the file if it doesn't exist, write to the end if it does
    with open("mtbench_moel_L4.jsonl", "a") as fout:
        fout.write(json.dumps(results) + "\n")
    
