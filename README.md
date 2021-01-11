# nnprof 

## Introduction

nnprof is a profile tool for pytorch neural networks. 

### Features

* **multi profile mode**: nnprof support 4 profile mode: Layer level, Operation level, Mixed level, Layer Tree level. Please check below for detail usage. 
* **time and memory profile**: nnprof support both time and memory profile now. But since memory profile is first supported in pytorch 1.6, please use torch version >= 1.6 for memory profile.
* **support sorted by given key and show profile percent**: user could print table with percentage and sorted profile info using a given key,  which is really helpful for optimiziing neural network.

## Requirements

* Python >= 3.6
* PyTorch
* Numpy

## Get Started

### install nnprof
* pip install: 
```shell
pip install nnprof
```
* from source: 
```shell
python -m pip install 'git+https://github.com/FateScript/nnprof.git'

# or install after clone this repo
git clone https://github.com/FateScript/nnprof.git
pip install -e nnprof
```

### use nnprf

```python3
from nnprof import profile, ProfileMode
import torch
import torchvision

model = torchvision.models.alexnet(pretrained=False)
x = torch.rand([1, 3, 224, 224])

# mode could be anyone in LAYER, OP, MIXED, LAYER_TREE
mode = ProfileMode.LAYER

with profile(model, mode=mode) as prof:
    y = model(x)

print(prof.table(average=False, sorted_by="cpu_time"))
# table could be sorted by presented header.
```

Part of presented table looks like table below, Note that they are sorted by cpu_time.
```
╒══════════════════════╤═══════════════════╤═══════════════════╤════════╕
│ name                 │ self_cpu_time     │ cpu_time          │   hits │
╞══════════════════════╪═══════════════════╪═══════════════════╪════════╡
│ AlexNet.features.0   │ 19.114ms (34.77%) │ 76.383ms (45.65%) │      1 │
├──────────────────────┼───────────────────┼───────────────────┼────────┤
│ AlexNet.features.3   │ 5.148ms (9.37%)   │ 20.576ms (12.30%) │      1 │
├──────────────────────┼───────────────────┼───────────────────┼────────┤
│ AlexNet.features.8   │ 4.839ms (8.80%)   │ 19.336ms (11.56%) │      1 │
├──────────────────────┼───────────────────┼───────────────────┼────────┤
│ AlexNet.features.6   │ 4.162ms (7.57%)   │ 16.632ms (9.94%)  │      1 │
├──────────────────────┼───────────────────┼───────────────────┼────────┤
│ AlexNet.features.10  │ 2.705ms (4.92%)   │ 10.713ms (6.40%)  │      1 │
├──────────────────────┼───────────────────┼───────────────────┼────────┤
```

You are welcomed to try diffierent profile mode and more table format.

## Contribution

Any issues and pull requests are welcomed.

## Acknowledgement

Some thoughts of nnprof are inspired by  [torchprof](https://github.com/awwong1/torchprof) and [torch.autograd.profile](https://github.com/pytorch/pytorch/blob/749f8b78508c43f9e6331f2395a4202785068442/torch/autograd/profiler.py) .
Many thanks to the authors.
