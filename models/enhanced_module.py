import torch
import numpy as np
import torch.nn as nn

from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.graph_module import GraphModule


class EnhancedNet(nn.Module):
    def __init__(self, num_class=16, num_heading_bin=1, num_size_cluster=16, mode="normal", use_contextual_aggregation=False,
    input_feature_dim=24, num_proposal=10, num_locals=10, vote_factor=1, sampling="vote_fps", query_mode="center", graph_mode="edge_conv", 
    num_graph_steps=2, use_relation=True, graph_aggr="add", use_orientation=True, num_bins=6, use_distance=False):
        super().__init__()

        self.mode = mode # normal or gt
        self.use_contextual_aggregation = use_contextual_aggregation

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        #self.mean_size_arr = mean_size_arr
        #assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = 128 if mode == "gt" else num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.num_graph_steps = num_graph_steps
        self.use_orientation = use_orientation
        self.use_distance = use_distance

        # --------- PROPOSAL GENERATION ---------
        if self.mode != "gt": # if not using GT data
            # Backbone point feature learning
            self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)


    # NOTE direct access only during inference
    def forward(self, points, use_tf=True, use_rl=False, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################
        data_dict = {}

        if self.mode != "gt":

            data_dict["point_clouds"] = points
        
            # --------- HOUGH VOTING ---------
            data_dict = self.backbone_net(data_dict)


        return data_dict