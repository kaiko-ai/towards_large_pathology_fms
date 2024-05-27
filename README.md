# Towards Large-Scale Training of Pathology Foundation Models
This repository contains the official implementation of the research paper: _"Towards Large-Scale Training of Pathology Foundation Models"_<br>
[[`whitepaper`](https://arxiv.org/abs/2404.15217)]

## Pretrained models

<div align="center">

| Model     | BACH  | CRC   | MHIST | PCam/val | PCam/test |
|-----------|-------|-------|-------|----------|-----------|
| ViT-S/16  | 0.797 | 0.943 | 0.828 | 0.903    | 0.893     |
| ViT-S/8   | 0.834 | 0.946 | 0.832 | 0.897    | 0.887     |
| ViT-B/16	| 0.810 | 0.960 | 0.826 | 0.900    | 0.898     |
| ViT-B/8   | 0.865 | 0.956 | 0.809 | 0.913    | 0.921     |
| ViT-L/14  | 0.870 | 0.930 | 0.809 | 0.908    | 0.898     |

_Table I: Linear probing evaluation of FMs on patch-level downstream datasets repoting<br> averaged balanced accuracy. All results were generated using [_eva_](https://github.com/kaiko-ai/eva/tree/main)._

</div>

### Pre-trained backbones (via PyTorch Hub)

Use the code below to get started with the models:
```py
# pip install torch timm
import torch

vits16 = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vits16")
vits8 = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vits8")
vitb16 = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb16")
vitb8 = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitb8")
vitl14 = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitl14")
```

Here is an end-to-end example:
```py
import io

import requests
import torch
from PIL import Image
from torchvision.transforms import v2

IMAGE_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQc7_xZpGOfQT7sxKwf2w5lL4GAq6IX_CbTzP1NGeenzA&s"
"""A sample WSI patch."""

preprocessing = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(size=224),
        v2.CenterCrop(size=224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ]
)
"""Model pre-processing."""

model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vits16")
"""Vision FM model."""


image = Image.open(io.BytesIO(requests.get(IMAGE_URL).content))
image_tensor = preprocessing(image)
features = model(image_tensor.unsqueeze(0))
assert features.shape == torch.Size([1, 384])
```

## Citation

If you find this repository helpful in your research, please consider citing our paper:
```
@misc{ai2024largescale,
    title={Towards Large-Scale Training of Pathology Foundation Models}, 
    author={kaiko.ai and Nanne Aben and Edwin D. de Jong and Ioannis Gatopoulos and Nicolas Känzig and Mikhail Karasikov and Axel Lagré and Roman Moser and Joost van Doorn and Fei Tang},
    year={2024},
    eprint={2404.15217},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
