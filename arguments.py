import argparse
from email.policy import default
import torch


def get_args():
	parser = argparse.ArgumentParser()

	# General Arguments
	parser.add_argument('--seed', type=int, default=1,
						help='random seed (default: 1)')
	parser.add_argument('--auto_gpu_config', type=int, default=1)
	parser.add_argument('--total_num_scenes', type=str, default="auto")
	parser.add_argument('-n', '--num_processes', type=int, default=5,
						help="""how many training processes to use (default:5)
								Overridden when auto_gpu_config=1
								and training on gpus""")
	parser.add_argument('--num_processes_per_gpu', type=int, default=6)
	parser.add_argument('--num_processes_on_first_gpu', type=int, default=1)
	parser.add_argument('--eval', type=int, default=0,
						help='0: Train, 1: Evaluate (default: 0)')
	parser.add_argument('--num_training_frames', type=int, default=10000000,
						help='total number of training frames')
	parser.add_argument('--num_eval_episodes', type=int, default=200,
						help="number of test episodes per scene")
	parser.add_argument('--num_train_episodes', type=int, default=100,
						help="""number of train episodes per scene
								before loading the next scene""")
	# GPU Configuration
	parser.add_argument('--no_cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument("--sim_gpu_id", type=str, default="4",
						help="gpu id on which scenes are loaded")
	
	parser.add_argument("--sem_gpu_id_list", type=str, default="4",
                        help="""gpu id list for semantic models,
                                -1: same as sim gpu, -2: cpu""")

	parser.add_argument("--policy_gpu_id", type=str, default="cuda:5",
						help="""policy gpu id for policy""")

	# Module Configuration
	parser.add_argument("--backbone_2d", type=str, default="rednet",
						help="""2d_backbone  maskrcnn/rednet""")
	parser.add_argument('--deactivate_klmap', action='store_true', default=False, 
						help="""deactivate KL divergency map True/False""")
	parser.add_argument('--deactivate_entropymap', action='store_true', default=False, 
						help="""deactivate entropy map True/False""")
	parser.add_argument('--deactivate_traphelper', action='store_true', default=False, 
						help="""deactivate trap helper True/False""")

	# Logging, loading models, visualization
	parser.add_argument('--log_interval', type=int, default=10,
						help="""log interval, one log per n updates
								(default: 10) """)
	parser.add_argument('--save_interval', type=int, default=1,
						help="""save interval""")
	parser.add_argument('-d', '--dump_location', type=str, default="./tmp/",
						help='path to dump models and log (default: ./tmp/)')
	parser.add_argument('--exp_name', type=str, default="exp1",
						help='experiment name (default: exp1)')
	parser.add_argument('--save_periodic', type=int, default=500000,
						help='Model save frequency in number of updates')



	parser.add_argument('--load_explore', type=str, default="0",
						help="""model path to load,
								0 to not reload (default: 0)""")

	parser.add_argument('--load_identify', type=str, default="0",
						help="""model path to load,
								0 to not reload (default: 0)""")


	parser.add_argument('-v', '--visualize', type=int, default=0,
						help="""1: Render the observation and
								   the predicted semantic map,
								2: Render the observation with semantic
								   predictions and the predicted semantic map
								(default: 0)""")
	parser.add_argument('--print_images', type=int, default=0,
						help='1: save visualization as images')

	# Environment, dataset and episode specifications
	parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
						help='Frame width (default:640)')
	parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
						help='Frame height (default:480)')
	parser.add_argument('-fw', '--frame_width', type=int, default=160,
						help='Frame width (default:160)')
	parser.add_argument('-fh', '--frame_height', type=int, default=120,
						help='Frame height (default:120)')
	parser.add_argument('-el', '--max_episode_length', type=int, default=500,
						help="""Maximum episode length""")

	# dataset

	parser.add_argument("--dataset", type=str, default="hm3d", 
					help="path to config yaml containing task information")



	parser.add_argument("--task_config", type=str,
						default="tasks/challenge_objectnav2022.local.rgbd.yaml",
						help="path to config yaml containing task information")



	parser.add_argument('--num_sem_categories', type=int, default=22,
						help = "number of semantic categories, 22 for mp3d/hm3d and 16 for gibson")




	# end dataset ==============================================
	parser.add_argument("--split", type=str, default="train",
						help="dataset split (train | val | val_mini) ")
	parser.add_argument('--camera_height', type=float, default=0.88,
						help="agent camera height in metres")
	parser.add_argument('--hfov', type=float, default=79.0,
						help="horizontal field of view in degrees")
	parser.add_argument('--turn_angle', type=float, default=30,
						help="Agent turn angle in degrees")
	parser.add_argument('--min_depth', type=float, default=0.5,
						help="Minimum depth for depth sensor in meters")
	parser.add_argument('--max_depth', type=float, default=5.0,
						help="Maximum depth for depth sensor in meters")
	parser.add_argument('--success_dist', type=float, default=1.0,
						help="success distance threshold in meters")
	parser.add_argument('--floor_thr', type=int, default=50,
						help="floor threshold in cm")
	parser.add_argument('--min_d', type=float, default=1.5,
						help="min distance to goal during training in meters")
	parser.add_argument('--max_d', type=float, default=100.0,
						help="max distance to goal during training in meters")
	# parser.add_argument('--version', type=str, default="v1.1",
	#					 help="dataset version")
	parser.add_argument('--version', type=str, default="v1",
						help="dataset version")
	# Model Hyperparameters
	parser.add_argument('--agent', type=str, default="sem_exp")
	parser.add_argument('--lr', type=float, default=2.5e-5,
						help='learning rate (default: 2.5e-5)')
	parser.add_argument('--global_hidden_size', type=int, default=256,
						help='global_hidden_size')
	parser.add_argument('--eps', type=float, default=1e-5,
						help='RL Optimizer epsilon (default: 1e-5)')
	parser.add_argument('--alpha', type=float, default=0.99,
						help='RL Optimizer alpha (default: 0.99)')
	parser.add_argument('--gamma', type=float, default=0.99,
						help='discount factor for rewards (default: 0.99)')
	parser.add_argument('--use_gae', action='store_true', default=False,
						help='use generalized advantage estimation')
	parser.add_argument('--tau', type=float, default=0.95,
						help='gae parameter (default: 0.95)')
	parser.add_argument('--entropy_coef', type=float, default=0.001,
						help='entropy term coefficient (default: 0.01)')
	parser.add_argument('--value_loss_coef', type=float, default=0.5,
						help='value loss coefficient (default: 0.5)')
	parser.add_argument('--max_grad_norm', type=float, default=0.5,
						help='max norm of gradients (default: 0.5)')
	parser.add_argument('--num_global_steps', type=int, default=25,
						help='number of forward steps in A2C (default: 5)')
	parser.add_argument('--ppo_epoch', type=int, default=4,
						help='number of ppo epochs (default: 4)')
	parser.add_argument('--num_mini_batch', type=str, default="auto",
						help='number of batches for ppo (default: 32)')
	parser.add_argument('--clip_param', type=float, default=0.2,
						help='ppo clip parameter (default: 0.2)')
	parser.add_argument('--use_recurrent_global', type=int, default=0,
						help='use a recurrent global policy')
	parser.add_argument('--num_local_steps', type=int, default=20,
						help="""Number of steps the local policy between each global step""")
	parser.add_argument('--reward_coeff', type=float, default=0.1,
						help="Object goal reward coefficient")
	parser.add_argument('--intrinsic_rew_coeff', type=float, default=0.02,
						help="intrinsic exploration reward coefficient")

	parser.add_argument('--sem_pred_lower_bound', type=float, default=0.75,
						help="Semantic prediction confidence threshold")



	# Mapping
	parser.add_argument('--global_downscaling', type=int, default=4)
	parser.add_argument('--vision_range', type=int, default=100)
	parser.add_argument('--map_resolution', type=int, default=5)
	parser.add_argument('--du_scale', type=int, default=1)
	parser.add_argument('--map_size_cm', type=int, default=4800)
	parser.add_argument('--map_point_size', type=int, default=4096) 

	# weight of semantic models
	parser.add_argument('--checkpt', type=str, default="./weight/rednet_semmap_mp3d_tuned.pth", 
						help='path to rednet models')

	parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
	parser.add_argument('--map_pred_threshold', type=float, default=1.0)
	parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
	parser.add_argument('--collision_threshold', type=float, default=0.20)

	# GL tree
	parser.add_argument('--point_size', type=int, default=512)
	parser.add_argument('--min_octree_threshold', type=float, default=4)
	parser.add_argument('--max_octree_threshold', type=float, default=15)
	parser.add_argument('--interval_size', type=float, default=20)

	parser.add_argument('--observation_window_size', type=int, default=4096 )


	# parse arguments
	args = parser.parse_args()

	args.cuda = not args.no_cuda and torch.cuda.is_available()

	if args.num_mini_batch == "auto":
		args.num_mini_batch = max(args.num_processes // 2, 1)
	else:
		args.num_mini_batch = int(args.num_mini_batch)


	return args