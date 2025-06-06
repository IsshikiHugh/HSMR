# Reconstructing Humans with a Biomechanically Accurate Skeleton

<a href="https://isshikihugh.github.io/HSMR/"><img src="https://img.shields.io/website?url=https%3A%2F%2Fisshikihugh.github.io%2FHSMR%2F&label=Project%20Page&up_message=Online&up_color=CAB7A5&down_message=Offline&down_color=%23FF3F4D&logo=googlechrome&logoColor=white"></a>
<a href="https://arxiv.org/abs/2503.21751"><img src="https://img.shields.io/badge/arXiv-2503.21751-%23B31C1C?logo=arxiv&logoSize=auto"></a>
<a href="https://www.cs.utexas.edu/~pavlakos/hsmr/resources/hsmr_suppmat.pdf"><img src="https://img.shields.io/badge/SupMat-PDF-%2347A141?logo=overleaf&logoColor=white"></a>
<a href="https://colab.research.google.com/drive/1RDA9iKckCDKh_bbaKjO8bQ0-Lv5fw1CB?usp=sharing"><img src="https://img.shields.io/badge/Demo-Open%20In%20Colab-blue?logo=googlecolab"></a>
<a href="https://huggingface.co/spaces/IsshikiHugh/HSMR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Open%20In%20HF-blue"></a>
<br>
<a href="#"><img src="https://img.shields.io/badge/HSMR%20Demo-Released-green?logo=github"></a>
<a href="#"><img src="https://img.shields.io/badge/HSMR%20Evaluation-Released-green?logo=github"></a>
<a href="#"><img src="https://img.shields.io/badge/HSMR%20Training-Released-green?logo=github"></a>
<a href="#"><img src="https://img.shields.io/badge/SKELify-Released-green?logo=github"></a>

<!-- ![](https://img.shields.io/github/stars/IsshikiHugh/HSMR) -->

<!-- Teaser Parts -->

![](https://isshikihugh.github.io/HSMR/assets/teaser_v2.png)

> [**Reconstructing Humans with a Biomechanically Accurate Skeleton**](https://isshikihugh.github.io/HSMR/)
> <br>
> [Yan Xia](https://scholar.isshikih.top),
> [Xiaowei Zhou](https://xzhou.me),
> [Etienne Vouga](https://www.cs.utexas.edu/~evouga/),
> [Qixing Huang](https://www.cs.utexas.edu/~huangqx/),
> [Georgios Pavlakos](https://geopavlakos.github.io/)
> <br>
> *CVPR 2025 (Oral)*


## 📢 News

- [2025.04.04] HSMR has been accepted as Oral paper at CVPR 2025! 🎉
- [2025.03.28] Release HSMR demo code, evaluation code and training code.

## ⚒️ Setup

1. [🌏 Environment Setup](./docs/SETUP.md#environment-setup)
2. [📦 Data Preparation](./docs/SETUP.md#data-preparation)

## 🚀 Demo & Quick Start

<!--
**[<img src="https://i.imgur.com/QCojoJk.png" width="30"> Google Colab demo](#) |
[<img src="https://s2.loli.net/2024/09/15/aw3rElfQAsOkNCn.png" width="20"> HuggingFace demo](#)**
-->

```shell
python exp/run_demo.py --help
```

Quick start with images:

```shell
# Folders wil be identified as image folders by default if `--input_type` is not specified.
python exp/run_demo.py --input_path "data_inputs/demo/example_imgs"
```

Quick start with videos:

```shell
# Single file wil be identified as a video by default if `--input_type` is not specified.
python exp/run_demo.py --input_path "data_inputs/demo/example_videos/gymnasts.mp4"
```

> Tips: Rendering skeleton meshes is pretty slow. For videos, adding `--ignore_skel` or decrease `--max_instances` could boost the speed. Check `lib/kits/hsmr_demo.py:parse_args()` for more details.

## 🧱 Reproducibility

For reproducing the results in the paper, please refer to [`docs/EVAL.md`](./docs/EVAL.md) and [`docs/TRAIN.md`](./docs/TRAIN.md).

We also provide the SKELify optimization pipeline, which optimizes SKEL parameters according to trustworthy 2D keypoints. Please refer to [`docs/OPTIM.md`](./docs/OPTIM.md) for more details.

## 🗓️ TODOs

- [x] Release Colab demo.
- [x] Release Huggingface demo.
- [x] Release SKELify pipeline.
- [x] Release training code & data.

## 📝 Citation

```bibtex
@inproceedings{xia2025hsmr,
  title={Reconstructing Humans with a Biomechanically Accurate Skeleton},
  author={Xia, Yan and Zhou, Xiaowei and Vouga, Etienne and Huang, Qixing and Pavlakos, Georgios},
  booktitle={CVPR},
  year={2025},
}
```

## 📜 Acknowledgement

Parts of the code are adapted from the following repos: [SKEL](https://github.com/MarilynKeller/SKEL), [4D-Humans/HMR2.0](https://github.com/shubham-goel/4D-Humans), [SMPLify-X](https://github.com/vchoutas/smplify-x), [SPIN](https://github.com/nkolot/SPIN), [ProHMR](https://github.com/nkolot/ProHMR), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [Detectron2](https://github.com/facebookresearch/detectron2), and [GVHMR](https://github.com/zju3dv/GVHMR).
