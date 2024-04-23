import torch
import torch.nn as nn
import torchvision.models
from torch.nn import functional as F
import numpy as np
import cv2
from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du
from utils.pointnet import PointNetEncoder
from utils.ply import write_ply_xyz, write_ply_xyz_rgb
from utils.img_save import save_semantic, save_KLdiv
from arguments import get_args
import os
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
from detectron2.utils import comm
from torch.distributions.categorical import Categorical as Categorical_IAM
from models.enhanced_module import EnhancedNet

scratch_cfg_file = '/home/cbl/alp/alp/configs/mask_rcnn/scratch_mask_rcnn_R_50_FPN_3x_syncbn.yaml'
class SelectStage(nn.Module):
    """Selects features from a given stage."""

    def __init__(self, stage: str = 'res5'):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return x[self.stage]


class DetectronResNet50(nn.Module):
    def __init__(self, downsample=True, distributed=True, device_id=0):
        super(DetectronResNet50, self).__init__()

        self.resnet_layer_size = 2048
        self.downsample = downsample

        cfg = get_cfg()
        cfg.merge_from_file(scratch_cfg_file)

        # use for feature encoder of policy training
        # random initialized weights and training mode
        cfg.MODEL.DEVICE = "cuda:{}".format(device_id)
        cfg.MODEL.RESNETS.NORM = 'BN'

        mask_rcnn = build_model(cfg)

        self.cnn = nn.Sequential(mask_rcnn.backbone.bottom_up, # resnet50 from mask_rcnn for multiple gpus
                                    SelectStage('res5'),
                                    # torch.nn.AdaptiveAvgPool2d(1), # res5 has shape bsz x 2048 x 8 x 8
                                    )
        self.pooling_layer = torch.nn.AdaptiveAvgPool2d(1)
        # input order and normalization
        self.input_order = cfg.INPUT.FORMAT
        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)

        # model is trainable
        for param in self.cnn.parameters():
            assert param.requires_grad

        """if comm.is_main_process() and pretrained:
            # sanity check: print model architecture
            print("Load resnet weight from {}".format(cfg.MODEL.WEIGHTS))
            print("Input channel order: {}".format(self.input_order))
            print("Normalize mean: {}".format(self.pixel_mean))
            print("Normalize std: {}".format(self.pixel_std))"""


    def forward(self, observations, pooling=True):
        device = observations.device
        self.pixel_mean = self.pixel_mean.to(device)
        self.pixel_std = self.pixel_std.to(device)

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
        # observations from rollout buffer already query "rgb" key value item
        rgb_observations = observations.permute(0, 3, 1, 2).contiguous()

        # downsample for faster compute
        if self.downsample:
            rgb_observations = F.avg_pool2d(rgb_observations, 2)

        if self.input_order == "BGR":
            # flip into BGR order
            rgb_observations = torch.flip(rgb_observations, dims=(1,)).contiguous()
        else:
            assert self.input_order == "RGB"

        # normalize
        rgb_observations = (rgb_observations - self.pixel_mean) / self.pixel_std

        # resnet forward -> last layer repr
        resnet_output = self.cnn(rgb_observations)
        if pooling:
            resnet_output = self.pooling_layer(resnet_output)
            # flatten dimension
            resnet_output = resnet_output.view(resnet_output.shape[0], resnet_output.shape[1])
        return resnet_output

    def save(self, folder_path, step, prefix="none"):
        if prefix == "none":
            torch.save(self.cnn[0].state_dict(), os.path.join(folder_path, "resnet_{}.pth".format(int(step))))
        else:
            torch.save(self.cnn[0].state_dict(), os.path.join(folder_path, "{}_resnet_{}.pth".format(prefix, int(step))))


    @property
    def output_size(self):
        return self.resnet_layer_size


class Explore_Network(NNBase):
    """
    2D 探索网络
    """
    def __init__(self, input_map_shape, input_points_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=6):
        super(Explore_Network, self).__init__(
            recurrent, hidden_size, hidden_size) # init gru

        out_size = int(input_map_shape[1] / 16.) * int(input_map_shape[2] / 16.)
        
        self.map_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )
        self.map_compress = nn.Linear(out_size * 32, 512)
        
        self.point_encoder = PointNetEncoder(global_feat=True,  \
                                             channel = num_sem_categories + 5) # use PointNet obtain map feature
        point_Encoder_state_dict = torch.load('pretrained/PointNet.pth',
                                          map_location=lambda storage, loc: storage)
        self.point_encoder.load_state_dict(point_Encoder_state_dict)

        self.point_compress = nn.Linear(1024, 512)

        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.time_emb = nn.Embedding(500, 8)

        self.linear1 = nn.Linear(512 + 512 + 8 * 3, hidden_size)
        #self.linear1 = nn.Linear(512 + 8 * 3, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)

        self.train()

    def forward(self, inputs_map, input_points, rnn_hxs, masks, extras):
        
        map_x = self.map_conv(inputs_map)
        map_x = self.map_compress(map_x) # （bs, 512)
        
        points_x = self.point_encoder(input_points)
        points_x = self.point_compress(points_x) #(bs, 512)

        orien_emb = self.orientation_emb(extras[:, 0]) # (bs, 8)
        goal_emb = self.goal_emb(extras[:, 1]) # (bs, 8)
        time_emb = self.time_emb(extras[:, 2]) # (bs, 8)

        x = torch.cat((map_x, points_x, orien_emb, goal_emb, time_emb), 1) # (bs, 1048)
        #x = torch.cat((map_x, orien_emb, goal_emb, time_emb), 1) # (bs, 1048)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks) # (bs, 256)
        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Explore_Policy(nn.Module):

    def __init__(self, obs_map_shape, obs_points_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Explore_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 1:
            self.network = Explore_Network(
                obs_map_shape, obs_points_shape, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs_map, inputs_points, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs_map, inputs_points , rnn_hxs, masks)
        else:
            return self.network(inputs_map, inputs_points, rnn_hxs, masks, extras)

    def act(self, inputs_map, inputs_points, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs_map, inputs_points, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs_map, inputs_points, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



class Identify_Network(NNBase):
    
    def __init__(self, obs_shape, obs_points_shape, recurrent, hidden_size = 512, points_channel_num = 12, num_sem_categories = 6):

        super(Identify_Network, self).__init__(
            recurrent, hidden_size, hidden_size) # init GRU

        C, N = obs_points_shape # Channel, points_num e.g. (27, 4096)

        #self.point_enhanced_Encoder = EnhancedNet()

        self.point_Encoder = PointNetEncoder(global_feat=True,  channel=C) # use PointNet obtain Gemotry features
        #point_Encoder_state_dict = torch.load('pretrained/PointNet.pth',
        #                                  map_location=lambda storage, loc: storage)
        #self.point_Encoder.load_state_dict(point_Encoder_state_dict)

        #self.resnet50 = DetectronResNet50(device_id=torch.cuda.current_device())
        #resnet_state_dict = torch.load('pretrained/ResNet50.pth',
        #                                 map_location=lambda storage, loc: storage)
        #self.resnet50.load_state_dict(resnet_state_dict)

        #self.policy_net = nn.Sequential(
        #    nn.Linear(2048 + 1024 + 3 * 8, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, 256),
        #    nn.ReLU()
        #)

        #self.linear1 = nn.Linear(2048 + 1024 + 8 * 3, hidden_size)
        self.linear1 = nn.Linear(1024 + 8 * 3, hidden_size)
        #self.linear1 = nn.Linear(32 * 512 + 8 * 3, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)

        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.time_emb = nn.Embedding(500, 8)

        self.critic_mlp = nn.Linear(256, 1)
        self.train()

    def forward(self, input_obs, input_points, rnn_hxs, masks, extras):

        args = get_args()
        # input_points: (bs, 27, 4096)
        #obs_feature = self.resnet50(input_obs[:, :, :, :3])

        #points_feature = self.point_enhanced_Encoder(input_points.transpose(1,2))
        #print(points_feature.size())
        #points_feature = torch.flatten(points_feature, 1, 2)
        #print(points_feature.size())

        points_feature = self.point_Encoder(input_points) # (bs, 1024)
        orientation_emb = self.orientation_emb(extras[:, 0]) # (bs, 8)
        goal_emb = self.goal_emb(extras[:, 1]) # (bs, 8)
        time_effe_emb = self.time_emb(extras[:, 2]) # (bs, 8)

        #x = torch.cat((obs_feature, points_feature, orientation_emb, goal_emb, time_effe_emb), 1) # (bs, 1048)
        x = torch.cat((points_feature, orientation_emb, goal_emb, time_effe_emb), 1) # (bs, 1048)

        x = nn.ReLU()(self.linear1(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks) # (bs, 256)
        x = nn.ReLU()(self.linear2(x))

        #x1 = self.policy_net(x) # (2, 256)

        return self.critic_mlp(x).squeeze(-1), x, rnn_hxs



# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Identify_Policy(nn.Module):

    def __init__(self, obs_shape, obs_points_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Identify_Policy, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 1:
            self.network = Identify_Network(
                obs_shape, obs_points_shape, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, input_obs, inputs_points, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(input_obs, inputs_points, rnn_hxs, masks, extras)
        else:
            return self.network(input_obs, inputs_points, rnn_hxs, masks, extras)

    def act(self, input_obs, inputs_points, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(input_obs, inputs_points, rnn_hxs, masks, extras) # 调用forward

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, input_obs, inputs_points, rnn_hxs, masks, extras=None):
        value, _, _ = self(input_obs, inputs_points, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs_obs, inputs_points, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs_obs, inputs_points, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()
        # print(args.device)
        # exit(0)
        self.device = args.device
        self.dataset = args.dataset
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)




    def forward(self, obs, pose_obs, maps_last, poses_last, origins, observation_points, goal_cat_id, gl_tree_list, infos, wait_env, args):

        bs, c, h, w = obs.size() # c:28 bs:2 h:120 w:160
        depth = obs[:, 3, :, :] # (bs, h, w)

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale) # (bs, h, w, 3)


        point_cloud_t_3d = point_cloud_t.clone() # depth中重构点云


        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device) # 坐标系变换 (bs, h, w, 3)
        
        agent_view_t_3d = point_cloud_t.clone()


        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device) # (bs, h, w 3)

        # from matplotlib import pyplot as plt
        # fig = plt.figure()
        # ax1 = plt.axes(projection='3d')
        # ax1.set_xlabel('x', size=20)
        # ax1.set_ylabel('y', size=20)
        # ax1.set_zlabel('z', size=20)
        # ax1.scatter3D(agent_view_centered_t.cpu()[1, :, :, 0], agent_view_centered_t.cpu()[1, :, :, 1], agent_view_centered_t.cpu()[1, :, :, 2],
        #               cmap='Blues')
        # plt.show()

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.

        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:4+(self.num_sem_categories), :, :]
        ).view(bs, self.num_sem_categories, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3]) # (2, 3, 19200)

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3) # （bs, 23, 100, 100, 80)


        min_z = int(25 / z_resolution - min_h) # min_z: 13
        max_z = int((self.agent_height + 1) / z_resolution - min_h) # max_z: 25

        agent_height_proj = voxels[..., min_z:max_z].sum(4) # (bs, 23, 100, 100)
        all_height_proj = voxels.sum(4) # (bs, 23, 100, 100) voxels -> map

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        # agent_view size: (bs, 26, 240, 240)
        channel = c
        if args.dataset == 'mp3d':
            channel = channel - 2 # -2 including, entropy, goal
        agent_view = torch.zeros(bs, channel,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred


        
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])


        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True) #(bs, 26, 240, 240)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True) #(bs, 26, 240, 240)

        points_pose = current_poses.clone() #(bs, 3)
        points_pose[:, :2] =  points_pose[:, :2] + torch.from_numpy(origins[:, :2]).to(self.device).float()

        points_pose[:,2] =points_pose[:,2] * np.pi /180 
        points_pose[:,:2] = points_pose[:,:2] * 100

        goal_maps = torch.zeros([bs, 1, 240, 240],dtype=float)

        for e in range(bs):

            world_view_t = du.transform_pose_t2(
                agent_view_t_3d[e,...], points_pose[e,...].cpu().numpy(), self.device).reshape(-1,3) # (19200, 3) RGB

            # world_view_sem_t: (19200, 22)
            world_view_sem_t = obs[e, 4:4+(self.num_sem_categories), :, :].reshape((self.num_sem_categories), -1).transpose(0, 1)

            # filter 过滤点云
            non_zero_row_1 = torch.abs(point_cloud_t_3d[e,...].reshape(-1,3)).sum(dim=1) > 0 # (19200,)
            non_zero_row_2 = torch.abs(world_view_sem_t).sum(dim=1) > 0 # (19200,)
            non_zero_row_3 = torch.argmax(world_view_sem_t, dim=1) != self.num_sem_categories-1 # (19200,)

            non_zero_row = non_zero_row_1 & non_zero_row_2 & non_zero_row_3 # (19200,)
            # non_zero_row = np.ones(19200, dtype=np.bool)
            world_view_sem = world_view_sem_t[non_zero_row].cpu().numpy() # (num, 22)

            if world_view_sem.shape[0] < 50:
                continue

            world_view_label = np.argmax(world_view_sem, axis=1) #(1600,)


            world_view_rgb = obs[e, :3, :, :].permute(1,2,0).reshape(-1,3)[non_zero_row].cpu().numpy() #(1600, 3)
            world_view_t = world_view_t[non_zero_row].cpu().numpy() # (pixels_num, 3)

            # from world_view of current frame sample 512 points and every point has 8 neighbors
            if world_view_t.shape[0] >= 1024:  # 512
                indx = np.random.choice(world_view_t.shape[0], 1024, replace = False) #(512, )
            else:
                indx = np.linspace(0, world_view_t.shape[0]-1, world_view_t.shape[0]).astype(np.int32)
            #print(world_view_label[indx])
            gl_tree = gl_tree_list[e]
            gl_tree.init_points_node(world_view_t[indx]) # every point init a octree
            per_frame_nodes = gl_tree.add_points(world_view_t[indx], world_view_sem[indx], world_view_rgb[indx], world_view_label[indx], infos[e]['timestep'])
            scene_nodes = gl_tree.all_points()
            gl_tree.update_neighbor_points(per_frame_nodes)

            #sample_points_tensor = torch.tensor(gl_tree.sample_points())   # local map
            sample_points_tensor = torch.tensor((gl_tree.sliding_window_points(
                          infos[e]['timestep'], 16)))

            sample_points_tensor[:,:2] = sample_points_tensor[:,:2] - origins[e, :2] * 100
            sample_points_tensor[:, 2]  = sample_points_tensor[:, 2] - 0.88 * 100
            sample_points_tensor[:,:3] = sample_points_tensor[:,:3] / args.map_resolution

            observation_points[e] = sample_points_tensor.transpose(1, 0)


        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return translated, fp_map_pred, map_pred, pose_pred, current_poses, observation_points

# resnet = torchvision.models.resnet50(pretrained=True)
# fc_inputs = resnet.fc.in_features
# resnet.fc = nn.Sequential(
#     nn.Linear(fc_inputs, 256),
#     nn.ReLU(),
#     nn.Dropout(0.4),
#     nn.Linear(256, 2048)
# )
#
# resnet.eval()
# img = np.random.random((10, 4, 480, 640)) * 255
# img = torch.from_numpy(img).float()
# print(resnet(img).size())

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,\
            padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,\
            bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class QNet(nn.Module):

    def __init__(self, num_sem_categories, layers=[2, 2, 2, 2]):
        super(QNet, self).__init__()

        block = BasicBlock
        self.inplanes = 32
        self.dilation = 1

        self.conv1 = nn.Conv2d(num_sem_categories + 8, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=1)
        self.layer4 = nn.Sequential(
            self._make_layer(block, 16, layers[3], stride=1),
            Flatten()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class Goal_Oriented_Semantic_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 num_sem_categories=16):
        super(Goal_Oriented_Semantic_Policy, self).__init__(
            recurrent, hidden_size, hidden_size)

        #out_size = int(input_shape[1] / 16.) * int(input_shape[2] / 16.)
        '''
        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(num_sem_categories + 8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )
        '''
        self.main_1 = QNet(num_sem_categories)

        #self.linear1 = nn.Linear(out_size * 32 + 8 * 2, hidden_size)
        self.linear1 = nn.Linear(900 * 16 + 8 * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):     # input:(b, num_sem_categories + 8, 240, 240)
        #x = self.main(inputs)                              # x: (b, 7200)
        x = self.main_1(inputs)

        orientation_emb = self.orientation_emb(extras[:, 0])
        goal_emb = self.goal_emb(extras[:, 1])

        x = torch.cat((x, orientation_emb, goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 1:
            self.network = Goal_Oriented_Semantic_Policy(
                obs_shape, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical_IAM):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        self.device = logits.device
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)
        

class RL_Policy_IAM(nn.Module):
    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):
        super(RL_Policy_IAM, self).__init__()
        self.network = Goal_Oriented_Semantic_Policy(
                obs_shape, **base_kwargs)

        self.linear = nn.Linear(self.network.output_size, action_space.n)

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False, invalid_action_masks=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)

        actor_features = self.linear(actor_features) # 256 -> 4
        if invalid_action_masks is not None:
            dist = CategoricalMasked(logits = actor_features, masks = invalid_action_masks)
        else:
            dist = Categorical(logits=actor_features)

        action = dist.sample()

        action_log_probs = dist.log_prob(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None, invalid_action_masks=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        actor_features = self.linear(actor_features) # 256 -> 4
        if invalid_action_masks is not None:
            dist = CategoricalMasked(logits = actor_features, masks = invalid_action_masks)
        else:
            dist = Categorical(logits=actor_features)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs