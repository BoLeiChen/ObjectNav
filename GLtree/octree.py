import numpy as np
from queue import Queue
import time
from GLtree.interval_tree import RedBlackTree, Node, BLACK, RED, NIL
import random
from constants import color_palette_array
from matplotlib import cm
import math

class point3D:
    """
    one point corresponds to an one-level octree map
    """
    def __init__(self, point_coor, point_color, num_sem_categories):
        self.point_coor = point_coor
        self.point_color = point_color
        self.point_seg_list = []


        self.seg_prob_fused = np.ones(num_sem_categories, dtype=float)
        self.label_thres = 0.0

        self.kl_div = 0.0
        self.entropy = 0.0

        self.max_prob = 0.0

        self.label = -1
        self.branch_array = [None, None, None, None, None, None, None, None]
        self.branch_distance = np.full((8),15)
        self.frame_id = 0

    def add_point_seg(self, point_seg):
        
        activate_3d = True

        self.point_seg_list.append(point_seg) # add semantic

        if activate_3d is False :
            return
        
        if self.label == -1: #init
            self.seg_prob_fused = point_seg.reshape(-1)
        else: #update
            self.seg_prob_fused = np.maximum(self.seg_prob_fused, point_seg.reshape(-1))
            self.seg_prob_fused /= np.sum(self.seg_prob_fused) # Normalization
        
        self.label = np.argmax(self.seg_prob_fused)

        tmp_prob_distribution = self.seg_prob_fused
        tmp_prob_distribution[np.where(tmp_prob_distribution == 0)] = 1 # skip log 0
        self.entropy = - np.sum(self.seg_prob_fused * np.log(tmp_prob_distribution))



class GL_tree:

    def __init__(self, opt):
        self.opt = opt
        self.x_rb_tree = RedBlackTree(opt.interval_size)
        self.y_rb_tree = RedBlackTree(opt.interval_size)
        self.z_rb_tree = RedBlackTree(opt.interval_size)

        self.scene_node = set()

        self.num_sem_categories = opt.num_sem_categories

        self.observation_window = set()
        self.observation_window_size = opt.observation_window_size

    def reset_gltree(self):
        # clear gl_tree
        del self.x_rb_tree
        del self.y_rb_tree
        del self.z_rb_tree
        del self.scene_node

        self.x_rb_tree = RedBlackTree(self.opt.interval_size)
        self.y_rb_tree = RedBlackTree(self.opt.interval_size)
        self.z_rb_tree = RedBlackTree(self.opt.interval_size)
        self.scene_node = set()
        self.observation_window = set()

    def init_points_node(self, points):
        """
        初始化八叉树结点 UPDATE Global Tree
        """
        self.x_tree_node_list = []
        self.y_tree_node_list = []
        self.z_tree_node_list = []

        for p in range(points.shape[0]):
            # 确定 p 落在哪一个区间结点中
            x_temp_node = self.x_rb_tree.add(points[p,0]) # X-coordinate
            y_temp_node = self.y_rb_tree.add(points[p,1]) # Y-coordinate
            z_temp_node = self.z_rb_tree.add(points[p,2]) # Z-coordinate

            self.x_tree_node_list.append(x_temp_node) # evert pixel c_x in the image corresponding node
            self.y_tree_node_list.append(y_temp_node)
            self.z_tree_node_list.append(z_temp_node)

    def add_points(self, points, point_seg, points_color, points_label, frame_index):
        """UPDATE local Tree"""
        activate_3d = True
        
        per_image_node_set = set()

        for p in range(points.shape[0]):
            
            x_set_union = self.x_tree_node_list[p].set_list
            y_set_union = self.y_tree_node_list[p].set_list
            z_set_union = self.z_tree_node_list[p].set_list # delete the point with same position in each coordinate
            set_intersection = x_set_union[0] & y_set_union[0] & z_set_union[0] # P_i^x and P_i^y and P_i^z = P_i^xyz
            temp_branch = [None, None, None, None, None, None, None, None] # octree's 8 branches
            temp_branch_distance = np.full((8), self.opt.max_octree_threshold) # self.opt.max_octree_threshold=15
            is_find_nearest = False
            branch_record = set()
            list_intersection=list(set_intersection)
            random.shuffle(list_intersection)

            for point_iter in list_intersection:
                ## 在GL Tree中定位pi 所在的结点，若找到，则更新语义, 并直接返回对应的 local tree
                distance = np.linalg.norm(point_iter.point_coor - points[p,:])
                if distance < self.opt.min_octree_threshold:
                    is_find_nearest = True
                    if frame_index != point_iter.frame_id:
                        point_iter.add_point_seg(point_seg[p, :])
                        point_iter.frame_id=frame_index
                        if activate_3d is False:
                            point_iter.label = points_label[p]
                    per_image_node_set.add(point_iter)
                    break

                x = int(point_iter.point_coor[0] >= points[p, 0])
                y = int(point_iter.point_coor[1] >= points[p, 1])
                z = int(point_iter.point_coor[2] >= points[p, 2])
                branch_num= x * 4 + y * 2 + z

                if distance < point_iter.branch_distance[7-branch_num]:
                    branch_record.add((point_iter, 7 - branch_num, distance))

                    if distance < temp_branch_distance[branch_num]:
                        temp_branch[branch_num] = point_iter
                        temp_branch_distance[branch_num] = distance
                        
            # 若没有找到，则为点云构建一个新的 local tree
            if not is_find_nearest:
                new_3dpoint = point3D(points[p, :], points_color[p, :], self.num_sem_categories)
                new_3dpoint.add_point_seg(point_seg[p, :])
                new_3dpoint.frame_id = frame_index
                if activate_3d is False:
                    new_3dpoint.label = points_label[p]
                for point_branch in branch_record:
                    point_branch[0].branch_array[point_branch[1]] = new_3dpoint
                    point_branch[0].branch_distance[point_branch[1]] = point_branch[2]

                new_3dpoint.branch_array = temp_branch
                new_3dpoint.branch_distance = temp_branch_distance # record the distance between the point and the neighbor
                per_image_node_set.add(new_3dpoint) # add point3D

                ## add a new point p in Node n
                for x_set in x_set_union:
                    x_set.add(new_3dpoint)
                for y_set in y_set_union:
                    y_set.add(new_3dpoint)
                for z_set in z_set_union:
                    z_set.add(new_3dpoint)


        self.observation_window = self.observation_window.union(per_image_node_set) # 求并集 observation_window according to the current pose
        # print("self.observation_window size:", len(self.observation_window))
        self.scene_node = self.scene_node.union(per_image_node_set) # the total num of the point3D in the scene
        return per_image_node_set

    def all_points(self):
        return self.scene_node

    def sample_points(self):
        #print("The number of points in the latest 8 frames:", len(self.observation_window))
        if len(self.observation_window) > self.observation_window_size:
            remove_node_list = random.sample(self.observation_window, len(self.observation_window) - self.observation_window_size)
            for node in remove_node_list:
                self.observation_window.remove(node)

        observation_points = np.zeros((4096, self.num_sem_categories + 3 + 1 + 1)) # sample 4096 points
        # bs, point positions(3) + semantic_categories(self.num_sem_categories) + semantic consistency(1) + entropy(1)
        for i, node in enumerate(self.observation_window):
            observation_points[i,:3] = node.point_coor
            observation_points[i,3:3 + self.num_sem_categories] = node.seg_prob_fused # point clouds with semantic
            observation_points[i,-2] = node.kl_div # KL
            observation_points[i,-1] = node.entropy # entropy

        return observation_points

    def sliding_window_points(self, frame_id_, window_size_):
        #print("The number of points in the latest 8 frames:", len(self.observation_window))
        if frame_id_ <= window_size_ - 1:
            observation_points = self.sample_points()
        else:
            remove_node = []
            for node in self.observation_window:
                if node.frame_id <= frame_id_ - window_size_:
                    remove_node.append(node)
            for node in remove_node:
                self.observation_window.remove(node)
            if len(self.observation_window) > self.observation_window_size:
                remove_node_list = random.sample(self.observation_window, len(self.observation_window) - self.observation_window_size)
                for node in remove_node_list:
                    self.observation_window.remove(node)

            observation_points = np.zeros((4096, self.num_sem_categories + 3 + 1 + 1))
            for i, node in enumerate(self.observation_window):
                observation_points[i, :3] = node.point_coor

                observation_points[i, 3:3 + self.num_sem_categories] = node.seg_prob_fused
                observation_points[i, -2] = node.kl_div
                observation_points[i, -1] = node.entropy

        return observation_points

    # simple update node
    def update_neighbor_points(self, per_image_node_set):
        for node in per_image_node_set:
            temp_kl_div_max = node.kl_div
            temp_center_prob = node.seg_prob_fused
            temp_center_prob[np.where(temp_center_prob == 0)] = 1
            center_point_kl_prob = node.seg_prob_fused * np.log(temp_center_prob)

            for i in range(8):
                if node.branch_array[i] is not None:
                    #---- KL Divergence MAX ---#
                    temp_branch_prob = node.branch_array[i].seg_prob_fused
                    temp_branch_prob[np.where(temp_branch_prob == 0)] = 1
                    branch_point_kl_prob = node.seg_prob_fused * np.log(temp_branch_prob)
                    
                    branch_kl_div = np.sum(center_point_kl_prob - branch_point_kl_prob)
                    if branch_kl_div > temp_kl_div_max:
                        temp_kl_div_max = branch_kl_div
            node.kl_div = temp_kl_div_max
            
            
    def find_object_goal_points(self, node_set, goal_obj_id, threshold):
        """
        Traverse point in the observation window, find the points with goal label
        """
        goal_list = []
        for node in node_set:
            if node.label != goal_obj_id or node.seg_prob_fused[goal_obj_id] < threshold:
                continue
            count = 0
            for i in range(8) :
                if node.branch_array[i] is not None and node.branch_array[i].label == goal_obj_id:
                    count += 1 
            if goal_obj_id == 5 or goal_obj_id == 6:
                if count <= 6:
                    temp_fused = np.copy(node.seg_prob_fused)
                    for i in range(8):
                        if node.branch_array[i] is not None:
                            temp_fused = np.maximum(temp_fused, node.branch_array[i].seg_prob_fused)
                    temp_fused /= np.sum(temp_fused)
                    node.seg_prob_fused = temp_fused
                    node.label = np.argmax(node.seg_prob_fused)
                    continue
            else:
                if count <= 5:
                    temp_fused = np.copy(node.seg_prob_fused)
                    for i in range(8):
                        if node.branch_array[i] is not None:
                            temp_fused = np.maximum(temp_fused, node.branch_array[i].seg_prob_fused)
                    temp_fused /= np.sum(temp_fused)
                    node.seg_prob_fused = temp_fused
                    node.label = np.argmax(node.seg_prob_fused)
                    continue                
            goal_list.append(node)

        if len(goal_list) == 0:
            return None 

        observation_points = np.zeros((len(goal_list), 3)) # x,y,z  
        for i, node in enumerate(goal_list):
            observation_points[i] = node.point_coor

        return observation_points



            


        

            