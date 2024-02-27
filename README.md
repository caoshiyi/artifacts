## Install
```
git clone https://github.com/caoshiyi/FastMoE.git
cd FastMoE

pip install -e .
```

## Run tests
The first-time weights loading can take ~10min.
```
python -m fastmoe.serve.launch_server --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 --port 30000
cd benchmarks
python run_exp.py --port 30000
```