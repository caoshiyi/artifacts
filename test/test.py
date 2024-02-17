import argparse

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"

    response = requests.post(
        url + "/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 128,
            },
        },
    )
    print(response.json())

    response = requests.post(
        url + "/generate",
        json={
            "text": "The capital of France is Paris.\nThe capital of the United States is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 128,
            },
        },
    )
    print(response.json())
