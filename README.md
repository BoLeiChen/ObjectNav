## Setup
- *Dependeces*: We use earlier (`0.2.2`) versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/v0.2.2) and [habitat-lab](https://github.com/facebookresearch/habitat-lab/tree/v0.2.2). Other related depencese can be found in `requirements.txt`. 

- *Data (MatterPort3D)*: Please download the scene dataset and the episode dataset from [habitat-lab/DATASETS.md](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset). Then organize the files as follows:
```
ObjectNav/
  data/
    scene_datasets/
        mp3d/
    episode_datasets/
        objectnav_mp3d_v1/
```
The weight of our 2D backbone RedNet can be found in [Stubborn](https://github.com/Improbable-AI/Stubborn). Please place the RedNet model in the ObjectNav/weight folder.


## Training and Evaluating:

We provide scripts for training and evaluation. You can modify these parameters to customize them according to your specific requirements.

For training:
```
python main.py --auto_gpu_config 0 -n 6  --sem_gpu_id_list "1,2,3" --policy_gpu_id "cuda:0" --sim_gpu_id "1,2,3" --split train --backbone_2d "rednet" --task_config "tasks/challenge_objectnav2021.local.rgbd.yaml" --dataset "mp3d" --num_sem_categories 22 --deactivate_entropymap  -d ./tmp --exp_name exp_mp3d --save_periodic 100000 --num_train_episodes 100 --use_recurrent_global 1 -v 0 --global_downscaling 2
```
For evaluation:
```
python main.py --sem_gpu_id_list "1,2,3" --sem_gpu_id "1,2,3" --num_processes 11 --sim_gpu_id "1,2,3" --policy_gpu_id "cuda:0" --split val --backbone_2d "rednet" --eval 1 -d ./tmp --print_images 0 --exp_name eval_mp3d_1 --dataset "mp3d" --num_sem_categories 22 --task_config "tasks/challenge_objectnav2021.local.rgbd.yaml" --load_explore /folder/to/periodic_explore_xxxxx.pth --load_identify /folder/to//periodic_explore_xxxxx.pth --use_recurrent_global 1 -v 0 --global_downscaling 2
```
