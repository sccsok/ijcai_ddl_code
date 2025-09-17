# 🏆 IJCAI 2025 - Deepfake Detection Pipeline

This repository contains our solution for the **IJCAI 2025 Workshop on Deepfake Detection, Localization, and Interpretability**.  
🥉 Achieved **3rd Place** in the **Image Track**.

## 📁 Project Structure

 ```
project_root/
├── data_generate  # Scripts for dataset formatting and JSON generation
├── ddl_data # The training、 test、validation images
├── image_process/ # Scripts for face cropping and preprocessing
├── Lav/ # LAV model implementation and configs
├── Mesorch/ # Mesorch model implementation and configs
├── test_lav.py # Inference script for LAV model
├── test_mesorch.py # Inference script for Mesorch model
├── merge.py # Merge LAV and Mesorch outputs
├── get_results.bash # The test script to get the final results
├── results/ # Stores intermediate and final results
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)
 ```
 
## 🔧 Environment Setup

- Python >= 3.8  
- (Recommended) Create a virtual environment:

    ```bash
    conda create -n ijcai_ddl python=3.8.*
    conda activate ijcai_ddl
    pip install -r requirements.txt
    ```

## 🚀 Data Preparation

- Prepare the image folder structure as shown below:

    ```
    ddl_data
        ├── train
            ├── real
            ├── fake
                ├── xxx.png
                └── ...
            └── mask
                ├── xxx.png
                └── ...
        ├── valid
            ├── real
            ├── fake
                ├── xxx.png
                └── ...
            └── mask
                ├── xxx.png
                └── ...
        └── test
            ├── xxx.png
            └── ...
    ```
- Expand the training data. This will generate a ```self``` folder under ddl_data.

    NOTE: The fake images we generated is on [Cloud Drive](https://pan.quark.cn/s/350e8743701f), you can download and put them into  ```ijcai_ddl_code/ddl_data/self```.

    You can also generate fake images using the following open source repositories. Note taht we only used the extracted face images with resolution larger than 512 to generate fake images (except for **Random combination**).

    The generated-image folder structure as shown below:

        ddl_data
        ├── self
            ├──Method1
                ├── fake
                    ├── xxx.png
                    └── ...
                └── mask
                    ├── xxx.png
                    └── ...
            ├──Method2
                ├── fake
                    ├── xxx.png
                    └── ...
                └── mask
                    ├── xxx.png
                    └── ...
            └── ...
    
    The details of self-constructed datasets are shown in Table 1.
    
    Table 1: Supplementary Data Registration for Deepfake Detection Model Training
    
	| Model Type        | Method              | Forgery Types           | Fake/Mask Image |
	|-------------------|---------------------|--------------------------|------------------|
	| Image Edit        | SBIs                | FaceSwap                 | 18135           |
	|                   | Random combination  | FaceSwap                 | 17728           |
	| GAN               | SimSwap             | FaceSwap                 | 14999           |
	|                   | MaskFaceGAN         | Face Attribute Editing   | 14999           |
	|                   | FaceDancer          | FaceSwap                 | 20000           |
	| Diffusion Model   | BELM                | Diffusion Inversion      | 14674           |
	|                   | SD-inpainting       | Inpainting               | 18347           |

    1. Image Edit: SBIs and Random combination

        1. [SBIs](https://arxiv.org/abs/2204.08376), [Github](https://github.com/mapooon/SelfBlendedImages) <br><br>

        2. Random combination

        You can reproduce the generation process of **Random combination** by running ```ijcai_ddl_code/data_generate/paste_random/generate_face.py``` and ```ijcai_ddl_code/data_generate/paste_random/generate_random.py```. 
        
        The former is to perform coarse-grained face splicing through landmarks, while the latter is to randomly paste faces into the target image. <br><br>

    2. GAN: Simswap, MaskFaceGAN and Facedancer

        1. [Simswap](https://arxiv.org/pdf/2106.06340v1.pdf), [Github](https://github.com/neuralchen/SimSwap) <br><br>

        2. [MaskFaceGAN](https://arxiv.org/abs/2103.11135), [Github](https://github.com/MartinPernus/MaskFaceGAN) <br><br>

        3. [Facedancer](https://openaccess.thecvf.com/content/WACV2023/html/Rosberg_FaceDancer_Pose-_and_Occlusion-Aware_High_Fidelity_Face_Swapping_WACV_2023_paper.html), [Github](https://github.com/felixrosberg/FaceDancer) <br><br>
        
    3. Diffusion: BELM and SD-inpainting

        1. [BELM](https://arxiv.org/abs/2410.07273), [Github](https://github.com/zituitui/BELM) <br><br>

        2. SD-inpainting, [Github](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

        You can reproduce the generation process of SD-inpainting following ```ijcai_ddl_code/data_generate/SDXL-inpainting/generate.py```.

        Download [stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) weights. <br><br>

- Processing training and test sets.

    1. Extracting faces for the **Lav** model.

    ```args.root``` is the path to the training dataset, ```args.out``` is the path to save the processed images.

    ```{args.out}/face_locations.json``` is to save all face locations for test images used in the **Lav** model validation.

    Download the [pritrained model](https://drive.google.com/file/d/1G82GU7uMw11xy3fGUKfdIzk09VV0qpb_/view?usp=sharing) of ```RetinaFace-Resnet50-fixed.pth``` and put it in ```ijcai_ddl_code/image_process/extract_face```

    ```bash
    cd ijcai_ddl_code
    cd image_process/extract_face

    # For training dataset
	python training_image_process.py \
    --root ../../ddl_data/train \
    --out ../../face/train \
    --gpu 0 \
    --worker 12
    
    # For test dataset
	python test_image_process.py \
    --root ../../ddl_data/test \
    --out ../../face \
    --gpu 0 \
    --worker 12
    ```

    2. Preparing the dataset loading files for **Lav** and **Mesorch**.

    ```args.face_root_dir``` is the path of all face images extracted from IJCAI_DDL training dataset (each must contain fake/, real/, mask/)
    
    ```args.self_root_dir``` is the path of self-collected datasets containing multiple subfolders (each contains fake/, real/, mask/).

    ```args.root_dir``` is the path of the orginal datasets containing train/valid/test folders.

    ```bash
    cd image_process

    # For training dataset of Lav
	python get_lav_data.py \
    --face_root_dir ../face/train \
    --self_root_dir ../ddl_data/self \
    --output_json ../Lav/datasets/ijcai_metadata_lav.json
    
    # For training dataset of Mesorch
	python get_mesorch_data.py \
    --flag train \
    --root_dir ../ddl_data \
    --self_root_dir ../ddl_data/self \
    --output_json ../Mesorch/datasets/ijcai_metadata_mesorch_train.json
    
    # For validation dataset of Mesorch
	python get_mesorch_data.py \
    --flag valid \
    --root_dir ../ddl_data \
    --output_json ../Mesorch/datasets/ijcai_metadata_mesorch_valid.json
    ```

## 📌 Training Process

- The training process of **Lav**.
    
    Download [pretrained model](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth) for **Xception** and put it in ```ijcai_ddl_code/Lav/training/checkpoints```.

    ```bash
    cd Lav/training
    bash train.bash
    ```
- The training Configuration for **Lav**

You can find more information in ```ijcai_ddl_code/Lav/training/config/detector/dual_stream.yaml```.

**Basic Training Configuration**

| Parameter   | Value |
| ----------- | ----- |
| Epochs      | 30    |
| Batch Size  | 76    |
| Num Workers | 16    |
| Resolution  | 299   |
| Manual Seed | 1024  |

**Optimizer and Learning Rate Scheduler**

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Optimizer     | Adam                            |
| Learning Rate | 5e-4                            |
| Beta          | (0.9, 0.999)                    |
| Epsilon       | 1e-8                            |
| Weight Decay  | 1e-5                            |
| LR Scheduler  | Cosine                          |

**Data Augmentation Parameters**

| Parameter        | Value     |
| ---------------- | --------- |
| Shift Limit      | 0.0625    |
| Scale Limit      | 0.2       |
| Rotate Limit     | 45°       |
| Blur Limit       | [3, 11]   |
| Noise Variance   | [10.0, 60.0] |
| Brightness/Contrast | [-0.3, 0.3] |
| JPEG Compression | [50, 100] |

**Model Architecture**

| Parameter        | Value     |
| ---------------- | --------- |
| Backbone         | Xception  |
| Pretrained Weights | `./checkpoints/xception-b5690688.pth` |



- The training process of **Mesorch**.

    Download [pretrained model](https://drive.google.com/file/d/1AKUdG34P3LoYMwvOp89WhVqVDqRj1q7G/view?usp=sharing) for **mixvit** and put it in ```ijcai_ddl_code/Mesorch/pretrained```.

    ```bash
    cd Mesorch/
    bash train_mesorch.sh
    ```

- The training Configuration for **Mesorch**
You can find more information in ```ijcai_ddl_code/Lav/training/config/detector/mesorch.yaml```.

**Basic Training Configuration**

| Parameter        | Value |
| ---------------- | ----- |
| Epochs           | 100   |
| Batch Size       | 24    |
| Num Workers      | 16    |
| Resolution       | 512   |
| Warmup Epochs    | 4     |

**Optimizer and Learning Rate Scheduler**

| Parameter               | Value                                      |
| ----------------------- | ------------------------------------------ |
| Optimizer               | AdamW                                      |
| Learning Rate           | 1e-4                                       |
| Beta                    | (0.9, 0.999)                               |
| Weight Decay            | 0.05                                       |
| warmup_epochs           | 2                                          |
| min_lr                  | 0.0000005                                  |
| LR Scheduler            | Cosine                                     |

**Data Augmentation Parameters**

| Parameter           | Value         |
| ------------------- | ------------- |
| Shift Limit         | 0.0625        |
| Scale Limit         | 0.1           |
| Rotate Limit        | 25°           |
| Blur Limit          | [3, 7]        |
| Noise Variance      | [10.0, 60.0]  |
| Brightness/Contrast | [-0.3, 0.3]   |
| JPEG Compression    | [50, 100]     |


**Model Architecture**

| Parameter        | Value     |
| ---------------- | --------- |
| Backbone         | mixvit  |
| Pretrained Weights | `./checkpoints/mit_b3.pth` |

## 🧪 Test Process

- The test process for the test dataset.

    Download [pretrained model](https://drive.google.com/file/d/1FfLkN1v-ZU6PvSYZHRfEVK-xrKIMlf2c/view?usp=sharing) for **Lav** and put it in ```ijcai_ddl_code/Lav/training/checkpoints```.

    Download [pretrained model](https://drive.google.com/file/d/18rqWLrNuUhDxZVCxvCzO4pFnfTZ2HsI4/view?usp=sharing) for **Mesorch** and put it in ```ijcai_ddl_code/Mesorch/ckpts```.

    ```bash
    bash get_results.bash
    ```

    1. We assume that you have already prepared and processed the training data according to the **Data Preparation**.

    2. If you need to test new images, you should prepare them with the special directory structure and construct input data format according to **Data Preparation**. Then modify the corresponding paths in ```get_results.bash```.

## 📄 Citation 

We sincerely thank the authors of **Mesorch** and **Lav** for their groundbreaking contributions to the field of image manipulation localization and deepfake detection. Their work has provided valuable insights and served as a strong foundation for our research.

### 🧠 Mesorch

```
@inproceedings{zhu2025mesoscopic,
  title={Mesoscopic insights: orchestrating multi-scale \& hybrid architecture for image manipulation localization},
  author={Zhu, Xuekang and Ma, Xiaochen and Su, Lei and Jiang, Zhuohang and Du, Bo and Wang, Xiwen and Lei, Zeyu and Feng, Wentao and Pun, Chi-Man and Zhou, Ji-Zhe},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={11022--11030},
  year={2025}
}
```

📄 [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/33198/35353)  

🔗 [GitHub Repository](https://github.com/scu-zjz/Mesorch)


### 🧠 Lav

```
@inproceedings{shuai2023locate,
  title={Locate and verify: A two-stream network for improved deepfake detection},
  author={Shuai, Chao and Zhong, Jieming and Wu, Shuang and Lin, Feng and Wang, Zhibo and Ba, Zhongjie and Liu, Zhenguang and Cavallaro, Lorenzo and Ren, Kui},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7131--7142},
  year={2023}
}
```

📄 [Paper](https://arxiv.org/pdf/2309.11131) 

🔗 [GitHub Repository](https://github.com/sccsok/Locate-and-Verify)

## 🙏 Acknowledgment

We sincerely thank our collaborators, data annotators, and supporters of this research project.  


