# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import os
import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
# from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat import Env, RLEnv, VectorEnv, make_dataset

from agents.sem_exp import Sem_Exp_Env_Agent
from agents.sem_exp_gibson import Sem_Exp_Gibson_Agent
from .objectgoal_env import ObjectGoal_Env
from .objectgoal_gibson_env import ObjectGoal_Gibson_Env
from .utils.vector_env import VectorEnv


def make_env_fn(args, config_env, rank):
    # 自定义的环境
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()

    if args.dataset == "gibson":
        if args.agent == "sem_exp":
            env = Sem_Exp_Gibson_Agent(args=args, rank=rank,
                                    config_env=config_env,
                                    dataset=dataset
                                    )
        else:
            env = ObjectGoal_Gibson_Env(args=args, rank=rank,
                                 config_env=config_env,
                                 dataset=dataset
                                 )
    else:
        if args.agent == "sem_exp":
            env = Sem_Exp_Env_Agent(args=args, rank=rank,
                                    config_env=config_env,
                                    dataset=dataset
                                    )
        else:
            env = ObjectGoal_Env(args=args, rank=rank,
                                 config_env=config_env,
                                 dataset=dataset
                                 )

    env.seed(rank)
    return env


def _get_scenes_from_folder(content_dir):
    #scene_dataset_ext = ".glb.json.gz"
    scene_dataset_ext = ".json.gz"

    scenes = []
    for filename in os.listdir(content_dir):

        if filename.endswith(scene_dataset_ext):
            #scene = filename[: -len(scene_dataset_ext) + 4]
            scene = filename[: -len(scene_dataset_ext)]
            scenes.append(scene)

    scenes.sort()
    return scenes


def construct_envs(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    # basic_config.DATASET.DATA_PATH = \
    #     basic_config.DATASET.DATA_PATH.replace("v1", args.version)
    # basic_config.DATASET.EPISODES_DIR = \
    #     basic_config.DATASET.EPISODES_DIR.replace("v1", args.version)
    basic_config.freeze()

    scenes = basic_config.DATASET.CONTENT_SCENES
    print("scenes1", scenes)
    if "*" in basic_config.DATASET.CONTENT_SCENES:
        content_dir = os.path.join(basic_config.DATASET.EPISODES_DIR.format(
            split=args.split), "content")
        scenes = _get_scenes_from_folder(content_dir)

    print("scenes2", scenes)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1


    # gpu_visible_devices = [2,2,2,2,2,3,3,3,3,3]
    # gpu_visible_devices = [1,1,1,1,1,3,3,3,3,3]
    gpu_visible_devices = [int(str_gpu_id) for str_gpu_id in args.sim_gpu_id.split(",")]

    print("Scenes per thread:")
    for i in range(args.num_processes):

        print("iter: ",str(i))

        config_env = cfg_env(config_paths=["envs/habitat/configs/"
                                           + args.task_config])
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                sum(scene_split_sizes[:i]):
                sum(scene_split_sizes[:i + 1])
            ]

            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # config_env.DATASET.CONTENT_SCENES = scenes[:2]
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


            print("Thread {}: {}".format(i, config_env.DATASET.CONTENT_SCENES))


        gpu_id = gpu_visible_devices[int(i%len(gpu_visible_devices))]


        print(">>>>>>>>>>>>", gpu_id)
                 
        # gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        # print("<<<<<<<<<<<", gpu_id)


        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")
        # agent_sensors.append("SEMANTIC_SENSOR")

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        # Reseting episodes manually, setting high max episode length in sim
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 10000000
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = args.min_depth
        config_env.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = args.max_depth
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]


#   ITERATOR_OPTIONS:
#     GROUP_BY_SCENE: True
#     NUM_EPISODE_SAMPLE: 1
#     SHUFFLE: False



        # config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.env_frame_width
        # config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.env_frame_height
        # config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
        # config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = \
        #     [0, args.camera_height, 0]

        config_env.SIMULATOR.TURN_ANGLE = args.turn_angle
        config_env.DATASET.SPLIT = args.split
        config_env.DATASET.DATA_PATH = \
            config_env.DATASET.DATA_PATH.replace("v1", args.version)
        config_env.DATASET.EPISODES_DIR = \
            config_env.DATASET.EPISODES_DIR.replace("v1", args.version)


        config_env.freeze()
        env_configs.append(config_env)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes))
            )
        ),
    )
    # exit(0)
    return envs
