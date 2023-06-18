NDDR-CNN
=============

#### - NDDR-CNN
  - A novel approach to improve the performance of multi-task convolutional neural networks (CNNs) through layerwise feature fusing and neural discriminative dimensionality reduction.
  - It leverages the idea of layerwise feature fusing, where features from different layers of the network are combined to obtain a more comprehensive representation. This allows the network to capture both low-level and high-level features relevant to each task.
  
#### - Neural Discriminative Dimensionality Reduction (NDDR) technique
  - NDDR learns a discriminative subspace by minimizing the within-class variations and maximizing the between-class separability.
  - This dimensionality reduction step helps to reduce the redundancy in the fused features and enhance their discriminative capabilities.
  
#### - Architecture of NDDR-CNN

<img src="https://github.com/qbxlvnf11/yolo-series/assets/52263269/96cd7379-2ba5-4370-85d8-22e8c74107f4" width="75%"></img> 

Contents & Descriptions
=============

#### - Modify Pytorch NDDR-CNN code in [official repository](https://github.com/bhpfelix/NDDR-CNN-PyTorch) to implement multi-task learning of gender/age classification

#### - Notice! Not same as backbone networks, parameter settings used in [paper](https://arxiv.org/abs/1801.08297)
  
#### - Train & validate each single-task (gender/age classification)
  
  - Structure of gender/age classification model
    - Refer to 'resnet_model.py'
    - Backbone network: Resnet50 with average pooling
    - Head layer: Drop-out + FC
    - Single model 1: gender classifier
      - Backbone network + head layers
      - 2 class (male/female)
    - Single model 2: age classifier
      - Backbone network + head layers
      - 100 class (1 ~ 100)
  
  - First, train each single task model before Multi-Task Learning (MTL) of two tasks

#### - Multi-Task Learning (MTL) & validate NDDR-CNN network for gender/age classification
  
  - Structure of NDDR-CNN network 
    - Refer to 'resnet_model.py' & 'nddr_net.py'
    - Append NDDR-CNN layer to each stage composed with resnet layers of two single-tasks
      - Stage 1: resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool
      - Stage 2: resnet50.layer1
      - Stage 3: resnet50.layer2
      - Stage 4: resnet50.layer3
      - Stage 5: resnet50.layer4
    - Append heads of two single-tasks to last layer of backbone
      - Two outputs: gender (2 class), age (100 class)
  
  - Loss of MTL
    - Sum the task-specific losses of each single-task 
    - In order to prevent only one single-task from being optimized because scales of the each single-task's losses are different during training, NORMAL_FACTOR is multipluated to maintain loss balance of the two single-tasks
    - E.g. loss = loss_gender_classification + NORMAL_FACTOR * loss_age_classification
  
  - After loading two pretrained weights of single models, MTL is performed to optimize NDDR-CNN layer and fine-tune weights of existing layers

#### - Experiments and Studies of NDDR-CNN

  - Parameters setting: refer to 'resnet_nddr.yaml'
  
  - Performances
    - It was confirmed that both gender/age classification performance exceeded that of the existing single model
  
  <img src="https://github.com/qbxlvnf11/yolo-series/assets/52263269/88cb6e6b-17ce-44b6-9e25-8bdc9d5b9861" width="80%"></img>

  - Studies
    - In order to maximize performance, it is necessary to search for the NORMAL_FACTOR parameter used to adjust the balance of the two losses and the NDDR_FACTOR, FC8_WEIGHT_FACTOR, and FC8_BIAS_FACTOR parameters used to give a larger learning rate to the NDDR-CNN layer to make the magnitude of the NDDR layer weight larger.
    - When fusing other types of single tasks (e.g. gender classification and age regression), MTL performance did not exceed single-task.

Structures of Project Folders
=============

```
${ROOT}
            |   |-- train_multi_task_resnet.py
            |   |-- train_single_task_resnet.py
            |   |-- ...
            |   |-- ckpts
            |   |   |   |-- gender_classifier ckpt.pth
            |   |   |   |-- age_classifier ckpt.pth 
            |   |   |   |-- nddr ckpt.pth
            |   |-- datasets
            |   |   |   |-- imdb-clean-master
            |   |   |   |   |   |-- run_all.sh
            |   |   |   |   |   |-- create_imdb_clean_1024.py
            |   |   |   |   |   |-- csvs
            |   |   |   |   |   |   |   |-- imdb_train_new.csv
            |   |   |   |   |   |   |   |-- imdb_train_new_1024.csv
            |   |   |   |   |   |   |   |-- imdb_valid_new.csv
            |   |   |   |   |   |   |   |-- imdb_valid_new_1024.csv
            |   |   |   |   |   |   |   |-- imdb_test_new.csv
            |   |   |   |   |   |   |   |-- imdb_test_new_1024.csv
            |   |   |   |   |   |-- data
            |   |   |   |   |   |   |   |-- imdb
            |   |   |   |   |   |   |   |   |   |-- 00
            |   |   |   |   |   |   |   |   |   |-- 01
            |   |   |   |   |   |   |   |   |   |-- ...
            |   |   |   |-- nyu_v2
            |   |   |   |   |   |-- ...
```


Download & Build Dataset (IMDB-Clean, nyu_v2)
=============

#### - Downlaod IMDB-WIKI & Build IMDB-Clean
  - IMDB-Clean:  Cleaning the noisy IMDB-WIKI dataset using a constrained clustering method
  - IMDB-Clean builder Github page: clone this repository

https://github.com/yiminglin-ai/imdb-clean/tree/master

  - Downlaod IMDB-WIKI & build IMDB-Clean
  
```
bash run_all.sh
```

#### - Downlaod nyu_v2 dataset

https://github.com/bhpfelix/NDDR-CNN-PyTorch


Download Pretrained Weights
=============

#### - Download IMDB-Clean
  - Password: 1234

http://naver.me/GbcGLvjn


Docker Environments
=============

#### - Pull docker environment

```
docker pull qbxlvnf11docker/nddr_cnn_env
```

#### - Run docker environment

```
nvidia-docker run -it --gpus all --name nddr_cnn_env --shm-size=64G -p 8888:8888 -e GRANT_SUDO=yes --user root -v {nddr_cnn_root_folder}:/workspace/nddr_cnn -w /workspace/nddr_cnn qbxlvnf11docker/nddr_cnn_env bash
```


How to use
=============

#### - Single-Task Learning IMDB-Clean
  - Gender classification
  
  ```
  CUDA_VISIBLE_DEVICES=0 python train_single_task_resnet.py \
    --config-file configs/resnet_single_task.yaml \
    --train_mode gender_classifier
  ```
  
  - Age classification
  
  ```
  CUDA_VISIBLE_DEVICES=0 python train_single_task_resnet.py \
    --config-file configs/resnet_single_task.yaml \
    --train_mode age_classifier
  ```
  
#### - Multi-Task Learning IMDB-Clean 

  ```
  CUDA_VISIBLE_DEVICES=0 python train_multi_task_resnet.py \
    --config-file configs/resnet_nddr.yaml
  ```

#### - Multi-Task Learning nyu_v2

  ```
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-file configs/vgg16_nddr_shortcut_sing_custom.yaml
  ```


References
=============

#### - NDDR-CNN
```
@article{NDDR-CNN,
  title={NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction},
  author={Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, Alan L. Yuille},
  journal = {IEEE Conference on Computer Vision and Pattern Recognition, 2019},
  year={2019}
}
```

#### - NDDR-CNN Pytorch

https://github.com/bhpfelix/NDDR-CNN-PyTorch

#### - IMDB-WIKI

https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

#### - Gender/Age Classifier

https://github.com/siriusdemon/pytorch-DEX


Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com


