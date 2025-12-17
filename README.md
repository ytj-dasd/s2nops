# Novel Class Discovery for Point Cloud Segmentation: An Semantic Prior-Guided and Superpoint-Enhanced Approach

![Model Architecture](assets/1.png)
![Results](assets/2.png)

## Introduction


Authors: 
        [Tangjun Yao]，
        [Shaoming Zhang],
        [Yuhan Zhang],
        [Siyu Wu],
        [Changhong Xie],
        [Baoxin Feng]


## Installation

The code has been tested with Python 3.8, CUDA 11.3, pytorch 1.10.1 and pytorch-lighting 1.4.8. Any other version may require to update the code for compatibility.

### Conda
To run the code, you need to install:
- [Pytorch 1.10.1](https://pytorch.org/get-started/previous-versions/)
- [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine)
- [Pytorch-Lighting 1.4.8](https://www.pytorchlightning.ai) (be sure to install torchmetrics=0.7.2)
- [Scipy 1.7.3](https://scipy.org/install/)
- [Wandb](https://docs.wandb.ai/quickstart)

## Data preparation
```
./
├── 
├── ...
└── path_to_data_shown_in_yaml_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── superpoint/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   └── labels/ 
            |          ├── 000000.label
            |          ├── 000001.label
            |          └── ...
            └── ...
```


## Commands
### Training
python main_discover.py -s 0 --dataset WhuMLS --dataset_config config/whumls_dataset.yaml --downsampling=80000 --voxel_size=0.025 --batch_size=2 --num_workers 24 --num_heads=5 --overcluster_factor=3 --use_scheduler --epochs=150 --queue_start_epoch=10 --warmup_epochs=20 --adapting_epsilon_sk --use_uncertainty_queue --use_uncertainty_loss --uncertainty_percentile=0.3 

### Inference
python inference.py -s 0 --dataset WhuMLS --dataset_config config/whumls_dataset.yaml  --checkpoint checkpoints_134/epoch\=99-step\=105199.ckpt --save_predictions --output_dir results/


Checkpoints to be placed in the `clip_chkpts` folder are available for download in the official [OpenScene](https://github.com/pengsongyou/openscene) repository.


