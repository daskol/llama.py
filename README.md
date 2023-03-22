# llama.py

**llama.py** is a fork of [llama.cpp][1] which provides Python bindings to an
inference runtime for [LLaMA][2] model in pure C/C++.

## Description

The main goal is to run the model using 4-bit quantization on a laptop.

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
ln -s build/release/llama/cc/_llama.cpython-310-x86_64-linux-gnu.so llama
```

Obtain the original LLaMA model weights and place them in `data/model` directory.

```shell
python -m llama pull -m data/model/7B -s 7B
```

As model weights are successfully fetched, directory structure should look like below.

```
data/model
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── tokenizer_checklist.chk
└── tokenizer.model
```

Then one should convert the 7B model to ggml FP16 format.

```shell
python -m llama convert data/model/7B
```

And quantize the model to 4-bits.

```shell
python -m llama quantize data/model/7B
```

Then one can start Python interpreter and play with naked bindings.

```python
from llama._llama import *

nothreads = 8
model = LLaMA.load('./data/model/7B/ggml-model-q4_0.bin', 512, GGMLType.F32)
mem_per_token = model.estimate_mem_per_token(nothreads)
logits = model.apply(context, context_size, mem_per_token, nothreads)

token_id = sample_next_token(context, logits)

tokenizer = model.get_tokenizer()
tokenizer.decode(token_id)

```

Or run CLI interface.

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


[1]: https://github.com/ggerganov/llama.cpp
[2]: https://arxiv.org/abs/2302.13971
