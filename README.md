# transformer_engine_tf2

# Introduction
This is a WIP project on supporting transformer engine in TF. The idea is
inspired by the [PyTorch version](https://github.com/NVIDIA/TransformerEngine).

Now, it contains the PoC (Proof of Concept) code to demonstrate how it would
look like of the TE in TF. It only supports the Dense layer on a single H100
GPU.

Install:

```bash
pip install pybind11
pip install .
```

Sanity test:

```bash
python run_sanity.py
python run_demo.py
```

Pybind test:
```bash
python test_pybind.py
```

Unit test:
```
python tests/fp8_layers_test.py
```
