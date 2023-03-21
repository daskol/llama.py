# llama.py

Inference of [LLaMA](https://arxiv.org/abs/2302.13971) model in pure C/C++.

## Description

The main goal is to run the model using 4-bit quantization on a MacBook

- Plain C/C++ implementation without dependencies.
- Apple silicon first-class citizen - optimized via ARM NEON.
- AVX2 support for x86 architectures.
- Mixed F16 / F32 precision.
- 4-bit quantization support.
- Runs on the CPU.

## Usage

Build instruction follows.

```shell
cmake -S . -B build/release
cmake --build build/release
```

Obtain the original LLaMA model weights and place them in `models` directory.

Install Python dependencies
```shell
pip install torch numpy sentencepiece
```

Convert the 7B model to ggml FP16 format.
```shell
python convert-pth-to-ggml.py models/7B/ 1
```

Quantize the model to 4-bits.
```shell
python quantize.py 7B
```

### Memory/Disk Requirements

As the models are currently fully loaded into memory, you will need adequate
disk space to save them and sufficient RAM to load them. At the moment, memory
and disk requirements are the same.

| model | original size | quantized size (4-bit) |
|-------|---------------|------------------------|
| 7B    | 13 GB         | 3.9 GB                 |
| 13B   | 24 GB         | 7.8 GB                 |
| 30B   | 60 GB         | 19.5 GB                |
| 65B   | 120 GB        | 38.5 GB                |
