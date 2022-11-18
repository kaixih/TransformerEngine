# transformer_engine_tf2

# Introduction
This is a WIP project on supporting transformer engine in TF. The idea is
inspired by the [PyTorch version](https://github.com/NVIDIA/TransformerEngine).

Now, it contains the PoC (Proof of Concept) code to demonstrate how it would
look like of the TE in TF. It only supports the Dense layer on a single GPU.

```
pip install pydantic
nvcc cu_demo.cu -Xcompiler -fPIC -shared -o cu_demo.so -std=c++17 -lcublasLt -lcublas
python run_demo.py
```
