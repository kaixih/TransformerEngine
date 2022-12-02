# transformer_engine_tf2

# Introduction
This is a WIP project on supporting transformer engine in TF. The idea is
inspired by the [PyTorch version](https://github.com/NVIDIA/TransformerEngine).

Now, it contains the PoC (Proof of Concept) code to demonstrate how it would
look like of the TE in TF. It only supports the Dense layer on a single H100
GPU.

```
pip install pydantic pybind11
nvcc pywrap_transformer_engine.cu -Xcompiler -fPIC -shared -I /usr/local/lib/python3.8/dist-packages/numpy/core/include -I/usr/local/lib/python3.8/dist-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 -DEIGEN_MAX_ALIGN_BYTES=64 -I/usr/include/python3.8 -I/usr/local/lib/python3.8/dist-packages/pybind11/include -o _pywrap_transformer_engine.cpython-38-x86_64-linux-gnu.so -std=c++17 -L/usr/local/lib/python3.8/dist-packages/tensorflow/python -l:../libtensorflow_framework.so.2 -l:_pywrap_tensorflow_internal.so -lcublasLt -lcublas
python run_demo.py
```

Pybind test

```
pip install pybind11
nvcc pywrap_transformer_engine.cu -Xcompiler -fPIC -shared --expt-relaxed-constexpr -I /usr/local/lib/python3.8/dist-packages/numpy/core/include -I/usr/local/lib/python3.8/dist-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 -DEIGEN_MAX_ALIGN_BYTES=64 -I/usr/include/python3.8 -I/usr/local/lib/python3.8/dist-packages/pybind11/include -o _pywrap_transformer_engine.cpython-38-x86_64-linux-gnu.so -std=c++17 -L/usr/local/lib/python3.8/dist-packages/tensorflow/python -l:../libtensorflow_framework.so.2 -l:_pywrap_tensorflow_internal.so -lcublasLt -lcublas
python test_pybind.py
```
