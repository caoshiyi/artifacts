## Install
```
git clone https://github.com/caoshiyi/FastMoE.git
cd FastMoE

pip install -e .
```

## Run tests
```
python -m fastmoe.serve.launch_server --model-path huggyllama/llama-7b --port 30000
cd test
python test.py
```