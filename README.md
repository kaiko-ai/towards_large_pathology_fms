# Towards Large-Scale Training of Pathology Foundation Models
This repository contains the official implementation of the research paper: _"Towards Large-Scale Training of Pathology Foundation Models"_<br>
[[`whitepaper`](https://github.com/kaiko-ai/towards_large_pathology_fms/blob/main/towards_large_scale_training_of_pathology_foundation_models.pdf?raw=true)]

## Pretrained models

<div align="center">

| Model                                            | BACH  | CRC   | MHIST | PCam/val | PCam/test |
|--------------------------------------------------|-------|-------|-------|----------|-----------|
| ViT-S/16              | 0.797 | 0.943 | 0.828 | 0.903    | 0.893     |
| ViT-S/8               | 0.834 | 0.946 | 0.832 | 0.897    | 0.887     |
| ViT-B/16	             | 0.810 | 0.960 | 0.826 | 0.900    | 0.898     |
| ViT-B/8               | 0.865 | 0.956 | 0.809 | 0.913    | 0.921     |
| ViT-L/14              | 0.870 | 0.930 | 0.809 | 0.908    | 0.898     |

_Table I: Linear probing evaluation of FMs on patch-level downstream datasets repoting<br> averaged balanced accuracy. All results were generated using [_eva_](https://github.com/kaiko-ai/eva/tree/main)._

</div>

### Pre-trained backbones (via PyTorch Hub)
To be released.
