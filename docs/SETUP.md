# Installations


## Environment Setup

![](https://img.shields.io/badge/-Env%20had%20been%20tested%20on%3A-gray?style=flat-square&logoColor=white)
![ubt20.04](https://img.shields.io/badge/Ubuntu-20.04-orange?logo=ubuntu&logoColor=white)
![py](https://img.shields.io/badge/Python-3.8,%203.10-%23356F9F?logo=python&logoColor=white)
![cuda12.3](https://img.shields.io/badge/CUDA-12.3-%2376B600?logo=nvidia&logoColor=white)


```shell
cd /path/to/HSMR
git submodule update --init  # fetch thirdparty code
```

```shell
# Use conda to create environment.
conda create -n hsmr python=3.10
conda activate hsmr

# Alternatively, you can use venv.
python3 --version  # ensure 3.8 or higher
python3 -m venv .hsmr_env
source .hsmr_env/bin/activate
```


```shell
# Install Dependencies
pip install -r requirements.txt  # make sure torch version is aligned with $CUDA_HOME
pip install "git+https://github.com/facebookresearch/detectron2.git"
pip install "git+https://github.com/mattloper/chumpy"
pip install -e .

# Install SKEL from https://github.com/MarilynKeller/SKEL
pip install -e thirdparty/SKEL
```



> If you encounter any version conflicts with `requirements.txt`, we also provide `docs/requirements_py3.8.txt` with version-annotated dependencies.


## Data Preparation


- If you only want to try the demo, you can follow the instruction in [Quick Start](#quick-start).
- If you want to check more details of our methods, please refer to the [Advanced Setup](#advanced-setup).

> 😊 Feel free to post an issue if you encounter any problems with data.

### Quick Start


**1/3**
```shell
# Regressors
mkdir -p data_inputs/body_models
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/SMPL_to_J19.pkl' \
    -O data_inputs/body_models/SMPL_to_J19.pkl
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/J_regressor_SKEL_mix_MALE.pkl' \
    -O data_inputs/body_models/J_regressor_SKEL_mix_MALE.pkl
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/J_regressor_SMPL_MALE.pkl' \
    -O data_inputs/body_models/J_regressor_SMPL_MALE.pkl
```
**2/3**
```shell
# HSMR Model
mkdir -p data_inputs/released_models/
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/released_models/HSMR-ViTH-r1d1.tar.gz' \
    -O HSMR-ViTH-r1d1.tar.gz
tar -xzvf HSMR-ViTH-r1d1.tar.gz -C data_inputs/released_models/
rm HSMR-ViTH-r1d1.tar.gz
```
**3/3**
Download `skel_models_v1.1.zip` from https://skel.is.tue.mpg.de/login.php and unzip it.

```shell
mkdir -p data_inputs/body_models
mv /path/to/skel_models_v1.1 data_inputs/body_models/skel
```

> Check [SKEL Model section](#skel-model) for more information about SKEL model.


Now you can return [Demo Instructions](../README.md#-demo--quick-start).

### Overview for Advanced Setup

> 🟢/🔵/🟣 indicate that the item is must-required/evaluation-required-only/training-required-only.

- 💀 Human Body Models
    - 🟢 [SKEL Model](#skel-model)
    - 🟢 [Auxiliary Regressors](#aux-reg)
    - 🔵 [SMPL Model](#smpl-model)
- 🚩 Checkpoints
    - 🟢 [HSMR Checkpoints](#hsmr-ckpt)
    - 🟣 [ViTPose Backbone Checkpoints](#bcb-ckpt)
- 📚 Datasets
    - 🔵 [Evaluation Datasets](#eval-ds)
    - 🟣 [Training Datasets](#train-ds)

### Human Body Models

> Tips: click to expand/fold instructions.

<a id="skel-model"></a>
<details>
  <summary>💀 SKEL Model</summary>
  <hr>

1. Go to [SKEL Homepage > Download](https://skel.is.tue.mpg.de/download.php) and download "SKEL and BSM models". You are supposed to get `skel_models_v1.1.zip`. The inside content should be like:

    ```shell
    skel_models_v1.1
    ├── Geometry/...
    ├── bsm.osim
    ├── changelog_v1.1.1.txt
    ├── sample_motion/...
    ├── skel_female.pkl
    ├── skel_male.pkl
    └── tmp.osim
    ```

    ⚠️ **SKEL version matters!** If you had downloaded the SKEL model before, please make sure the whole version number is `v1.1.1`. You can find the model version in `changelog_v1.1.1.txt`. Any lower versions are **not compatible**.
2. Add `skel_models_v1.1` to the code base:

    ```shell
    mkdir -p data_inputs/body_models
    mv /path/to/skel_models_v1.1 data_inputs/body_models/skel
    ```

> For more information about the SKEL model itself, please refer [SKEL project page](https://skel.is.tue.mpg.de/) or [LearningHumans Jupyter notebook](https://github.com/IsshikiHugh/LearningHumans/blob/main/notebooks/SKEL_basic.ipynb).

  <hr>
</details>


<a id="aux-reg"></a>
<details>
  <summary>🛠️ Auxiliary Regressors</summary>
  <hr>

Download the necessary additional files from [HuggingFace](https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/) and put them to `data_inputs/body_models/`. Or you can simply run this:

```shell
mkdir -p data_inputs/body_models
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/SMPL_to_J19.pkl' \
    -O data_inputs/body_models/SMPL_to_J19.pkl
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/J_regressor_SKEL_mix_MALE.pkl' \
    -O data_inputs/body_models/J_regressor_SKEL_mix_MALE.pkl
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/body_models/J_regressor_SMPL_MALE.pkl' \
    -O data_inputs/body_models/J_regressor_SMPL_MALE.pkl
```

  <hr>
</details>


<a id="smpl-model"></a>
<details>
  <summary>🧑‍🦲 SMPL Model</summary>
  <hr>

1. Sign in [SMPLify Homepage > Downloads](https://smplify.is.tue.mpg.de/download.php) and click `SMPLIFY_CODE_V2.ZIP`. You will get `mpips_smplify_public_v2.zip`. After unzipping, you will get a folder like this:

    ```shell
    smplify_public
    ├── code
    │   ├── models
    │   │   ├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl  # 10-shape model, in size of 37MB
    │   │   └── ...
    │   └── ...
    └── ...
    ```

2. Add to the code base.

    ```shell
    mkdir -p data_inputs/body_models/smpl
    mv /path/to/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data_inputs/body_models/smpl/SMPL_NEUTRAL.pkl
    ```

> For more information about the SMPL model itself, please refer [SMPL project page](https://smpl.is.tue.mpg.de/) or [LearningHumans Jupyter notebook](https://github.com/IsshikiHugh/LearningHumans/blob/main/notebooks/SMPL_basic.ipynb).


  <hr>
</details>


### Checkpoints

<a id="hsmr-ckpt"></a>
<details>
  <summary>🚩 HSMR Checkpoint</summary>
  <hr>

1. Download `HSMR-ViTH-r1d1.tar.gz` from [HuggingFace](https://huggingface.co/IsshikiHugh/HSMR-data_inputs/tree/main/released_models), or you can simply run this:
    ```shell
    wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/released_models/HSMR-ViTH-r1d1.tar.gz' \
        -O HSMR-ViTH-r1d1.tar.gz
    ```
2. Add to the code base.

    ```shell
    mkdir -p data_inputs/released_models/
    tar -xzvf HSMR-ViTH-r1d1.tar.gz -C data_inputs/released_models/
    rm HSMR-ViTH-r1d1.tar.gz
    ```

  <hr>
</details>

<a id="bcb-ckpt"></a>
<details>
  <summary>🚩 ViTPose Backbone Checkpoint</summary>
  <hr>

Download `vitpose_backbone.pth` from [HuggingFace](https://huggingface.co/IsshikiHugh/HSMR-data_inputs/tree/main/backbone) and put them to `data_inputs/backbone/`.Or you can simply run this:

```shell
mkdir -p data_inputs/backbone/
wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/backbone/vitpose_backbone.pth' \
    -O data_inputs/backbone/vitpose_backbone.pth
```

  <hr>
</details>


### Datasets

<a id="eval-ds"></a>
<details>
  <summary>🩺 Evaluation Datasets</summary>
  <hr>

We evaluate our approach on multiple benchmarks. Specifically, we use the benchmarks used by [HMR2.0](https://github.com/shubham-goel/4D-Humans), including 2D datasets (COCO, LSP-EXTENDED, PoseTrack) and 3D datasets (3DPW, H36M). Moreover, we evaluate on the [MOYO](https://moyo.is.tue.mpg.de/) dataset.


1. Prepare the evaluation datasets used by HMR2.0:
    1. Download `hmr2_evaluation_data.tar.gz` from [DropBox](https://www.dropbox.com/scl/fi/kl79djemdgqcl6d691er7/hmr2_evaluation_data.tar.gz?rlkey=ttmbdu3x5etxwqqyzwk581zjl&e=1) or [HuggingFace](https://huggingface.co/IsshikiHugh/HSMR-data_inputs/tree/main/), or you can simply run this:
        ```shell
        wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/hmr2_evaluation_data.tar.gz' \
            -O hmr2_evaluation_data.tar.gz
        ```
    2. Prepare [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/), [H36M](http://vision.imar.ro/human3.6m/description.php), [COCO](https://cocodataset.org/#download), [LSP-EXTENDED](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip), and [PoseTrack](https://posetrack.net/) datasets. Check the "data structure" bellow to see how to organize the data. Only the images are required.
2. Prepare the MOYO evaluation dataset:
    1. Download `hsmr_evaluation_data.tar.gz` from [HuggingFace](https://huggingface.co/IsshikiHugh/HSMR-data_inputs/tree/main/), or you can simply run this:
        ```shell
        wget -c 'https://huggingface.co/IsshikiHugh/HSMR-data_inputs/resolve/main/hsmr_evaluation_data.tar.gz' \
            -O hsmr_evaluation_data.tar.gz
        ```
    2. Download MoYo images from [here](https://github.com/sha2nkt/moyo_toolkit).
3. Add to the code base.
    ```shell
    # Prepare meta data.
    tar -xzvf hmr2_evaluation_data.tar.gz -C data_inputs/
    rm hmr2_evaluation_data.tar.gz
    tar -xzvf hsmr_evaluation_data.tar.gz -C data_inputs/
    rm hsmr_evaluation_data.tar.gz
    ```
    ```shell
    # Prepare image data.
    mkdir -p data_inputs/datasets
    ln -s /path/to/3dpw      data_inputs/datasets/3dpw
    ln -s /path/to/h36m      data_inputs/datasets/h36m
    ln -s /path/to/coco      data_inputs/datasets/coco
    ln -s /path/to/hr-lspet  data_inputs/datasets/hr-lspet
    ln -s /path/to/posetrack data_inputs/datasets/posetrack
    ln -s /path/to/moyo      data_inputs/datasets/moyo
    ```

After that, the data structure should fit:

```shell
data_inputs
├── datasets  # The image files.
│   ├── 3dpw
│   │   └── imageFiles/*/*.jpg
│   ├── coco
│   │   └── val2017/*.jpg
│   ├── h36m
│   │   └── images/*.jpg
│   ├── hr-lspet
│   │   └── *.png
│   ├── posetrack
│   │   └─── posetrack2018/posetrack_data/images/*/*/*.jpg
│   └── moyo
│       ├── 220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a
│       │   └── YOGI_Cam_*/*.jpg
│       ├── 220923_yogi_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-b
│       │   └── YOGI_Cam_*/*.jpg
│       └── ...
├── hmr2_evaluation_data  # The packaged labels of standard benchmark datasets.
│   ├── 3dpw_test.npz
│   ├── coco_val.npz
│   ├── h36m_val_p2.npz
│   ├── hr-lspet_train.npz
│   └── posetrack_2018_val.npz
└── hsmr_evaluation_data  # The packaged labels of extra benchmark datasets.
    └── moyo_v2.npz
```

  <hr>
</details>


<a id="train-ds"></a>
<details>
  <summary>🏋 Training Datasets</summary>
  <hr>

1. Download training data parts through the commands below (or click the links in the table):
    ```shell
    mkdir -p data_inputs/hsmr_training_data
    wget "https://www.dropbox.com/scl/fi/tdnyxoufx8u3f7kcgruah/hsmr_training_data.part1.tar.gz?rlkey=92pm05qa8pimjhrpwipu56svn&st=id2dvek4&dl=1" \
        -O data_inputs/hsmr_training_data/hsmr_training_data.part1.tar.gz
    wget "https://www.dropbox.com/scl/fi/az8opeka0zyhl6mk0gj6p/hsmr_training_data.part2.tar.gz?rlkey=7iv6t1hl95ok6zuxp2nit29fr&st=5jepxumi&dl=1" \
        -O data_inputs/hsmr_training_data/hsmr_training_data.part2.tar.gz
    wget "https://www.dropbox.com/scl/fi/rd9uribnjvyj896cqbj7o/hsmr_training_data.part3.tar.gz?rlkey=b821r5qslbuqivqav8qvu8fee&st=n215nv19&dl=1" \
        -O data_inputs/hsmr_training_data/hsmr_training_data.part3.tar.gz
    wget "https://www.dropbox.com/scl/fi/pxm75g95nbkqxg3er8ozd/hsmr_training_data.part4.tar.gz?rlkey=7e5ftkbzre0smbxi2mtukijuk&st=0bje2g3n&dl=1" \
        -O data_inputs/hsmr_training_data/hsmr_training_data.part4.tar.gz
    wget "https://www.dropbox.com/scl/fi/3us8qbxra7v01marw52g3/hsmr_training_data.part5.tar.gz?rlkey=737cofc38z9imafk016n4lxmz&st=qowg65xy&dl=1" \
        -O data_inputs/hsmr_training_data/hsmr_training_data.part5.tar.gz
    wget "https://www.dropbox.com/scl/fi/2du8j0wll367mxmqk8u37/hsmr_training_data.part6.tar.gz?rlkey=cdz3trhq1eycyko0ahba5ojen&st=ke4a306z&dl=1" \
        -O data_inputs/hsmr_training_data/hsmr_training_data.part6.tar.gz
    wget "https://www.dropbox.com/scl/fi/kb2gv7z45d9ha3clof33j/hsmr_training_data.part7.tar.gz?rlkey=7sonxtqqhuctvzfrpecdnbpd1&st=lbv2oh5n&dl=1" \
        -O data_inputs/hsmr_training_data/hsmr_training_data.part7.tar.gz
    ```
    | file name | sha1sum | size |
    |---|---|---|
    |[hsmr_training_data.part1.tar.gz](https://www.dropbox.com/scl/fi/tdnyxoufx8u3f7kcgruah/hsmr_training_data.part1.tar.gz?rlkey=92pm05qa8pimjhrpwipu56svn&st=id2dvek4&dl=0)|`6675c3f5987186893c4bc5c8616a5ce743ce941d`|30G|
    |[hsmr_training_data.part2.tar.gz](https://www.dropbox.com/scl/fi/az8opeka0zyhl6mk0gj6p/hsmr_training_data.part2.tar.gz?rlkey=7iv6t1hl95ok6zuxp2nit29fr&st=5jepxumi&dl=0)|`dd62b3e86e8c8ae33436aa5e54d4e9ef27a64503`|41G|
    |[hsmr_training_data.part3.tar.gz](https://www.dropbox.com/scl/fi/rd9uribnjvyj896cqbj7o/hsmr_training_data.part3.tar.gz?rlkey=b821r5qslbuqivqav8qvu8fee&st=n215nv19&dl=0)|`97a98f70488fddff2ac7a4055b4a831039d95ac4`|39G|
    |[hsmr_training_data.part4.tar.gz](https://www.dropbox.com/scl/fi/pxm75g95nbkqxg3er8ozd/hsmr_training_data.part4.tar.gz?rlkey=7e5ftkbzre0smbxi2mtukijuk&st=0bje2g3n&dl=0)|`5ac1f308fe0c0e96db7c27308292ae03d2fd9d1f`|39G|
    |[hsmr_training_data.part5.tar.gz](https://www.dropbox.com/scl/fi/3us8qbxra7v01marw52g3/hsmr_training_data.part5.tar.gz?rlkey=737cofc38z9imafk016n4lxmz&st=qowg65xy&dl=0)|`b0898c9ef37f417c15589224199948476cfc3abe`|39G|
    |[hsmr_training_data.part6.tar.gz](https://www.dropbox.com/scl/fi/2du8j0wll367mxmqk8u37/hsmr_training_data.part6.tar.gz?rlkey=cdz3trhq1eycyko0ahba5ojen&st=ke4a306z&dl=0)|`c7cc0a94f439fd4614b2c97d62767bcc2332f2a2`|37G|
    |[hsmr_training_data.part7.tar.gz](https://www.dropbox.com/scl/fi/kb2gv7z45d9ha3clof33j/hsmr_training_data.part7.tar.gz?rlkey=7sonxtqqhuctvzfrpecdnbpd1&st=lbv2oh5n&dl=0)|`c62b5f96d0168a5ac0eacb772ecc3eb55c9b2756`|39G|
    - Alternatively, you can download the un-packaged webdataset tars from [HuggingFace Dataset](https://huggingface.co/datasets/IsshikiHugh/HSMR-TrainingData/tree/main).
2. Check the integrity of the downloaded files:
    ```shell
    sha1sum hsmr_training_data.part*.tar.gz
    ```
3. Unzip and add to the codebase.
    ```shell
    cd data_inputs/hsmr_training_data
    for file in hsmr_training_data.part*.tar.gz; do
        tar -xzvf "$file"
    done
    ```

  <hr>
</details>