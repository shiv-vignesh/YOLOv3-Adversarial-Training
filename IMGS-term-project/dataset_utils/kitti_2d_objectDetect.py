import os, open3d
import cv2 
import albumentations
import numpy as np
from time import time

from typing import List, Dict, Iterable

import torch
from torch.utils.data import Dataset
from imgaug import augmenters as iaa

from .enums import Enums

#monkey patch for iaa augmentation
np.bool = bool

class LidarPreprocessorUtils:
    
    def transform_lidar_points(self, point_cloud_array:np.array, calibration_dict:dict):

        # ignoring the last_dim as it corresponds to intensity.
        if point_cloud_array.shape[-1] > 3:
            point_cloud_array = point_cloud_array[:, :3] #(N, 3)
        
        #reshaping rectification matrix from (12,) to (3,3)
        r0_rect = calibration_dict['R0_rect'].reshape(3,3) #(3,3)
        r0_rect_homo = np.vstack([r0_rect, [0, 0, 0]]) #(4,3)
        r0_rect_homo = np.column_stack([r0_rect_homo, [0, 0, 0, 1]]) #(4,4)
        
        # reshaping projection_matrix from (12,) to (3,4)
        proj_mat = calibration_dict['P2'].reshape(3,4) 
        
        # reshaping Tr_velo_to_cam from (12,) to (3,4)
        v2c = calibration_dict['Tr_velo_to_cam'].reshape(3,4)
        v2c = np.vstack(
            (v2c, [0, 0, 0, 1])
        ) #(4,4)    
        
        p_r0 = np.dot(proj_mat, r0_rect_homo) # (3, 4)
        p_r0_rt = np.dot(p_r0, v2c) #(3, 4)
        
        point_cloud_array = np.column_stack(
            [point_cloud_array, np.ones((point_cloud_array.shape[0], 1))]
        ) # (N, 4)
        
        #(3, 4) dot (4, N) ---> (3, N) ---> (N, 3)
        p_r0_rt_x = np.dot(
            p_r0_rt, point_cloud_array.T
        ).T 
        
        # # The transformed coordinates are for LIDAR (u, v, z) to (u', v', z') in Image. Normalize by depth (z')
        p_r0_rt_x[:, 0] /= p_r0_rt_x[:, -1]
        p_r0_rt_x[:, 1] /= p_r0_rt_x[:, -1]

        depth = p_r0_rt_x[:, -1]        
        negative_depth_mask = depth < 0
        
        return p_r0_rt_x[:, :2], p_r0_rt_x[:, -1], negative_depth_mask
        
        # proj_points_2d = p_r0_rt_x[:, :2]
        # depth = p_r0_rt_x[:, -1]
        
        # nonzero_depth_mask = depth > 0
        # # proj_points_2d = proj_points[nonzero_depth_mask]
        
        # depth[nonzero_depth_mask] = 1e-5
        # # valid_depth = depth > 1e-5 
        
        # proj_points_2d[:, 0] /= depth
        # proj_points_2d[:, 1] /= depth
        
        # return proj_points_2d[:, :2], depth
    
    def voxel_downsampling_torch(self, lidar_point_cloud:torch.tensor, 
                                 voxel_size = 0.2,  # Voxel size
                                 num_points:int=50000):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        lidar_point_cloud = lidar_point_cloud.to(device)
        voxel_indices = torch.floor(lidar_point_cloud[:, :3] / voxel_size).int()  # (N, 3)

        # Step 2: Group points by voxel
        unique_voxels, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)

        # Step 3: Aggregate points within each voxel (centroid or random)
        num_voxels = unique_voxels.size(0)
        aggregated_points = []
        for voxel_idx in range(num_voxels):
            mask = inverse_indices == voxel_idx
            voxel_points = lidar_point_cloud[mask]
            centroid = voxel_points.mean(dim=0)  # Compute centroid
            aggregated_points.append(centroid)

        downsampled_points = torch.stack(aggregated_points, dim=0)  # (M, D), where M = num_voxels

        # Step 4: Subsample or pad to fixed number of points if required
        if num_points:
            num_current_points = downsampled_points.size(0)
            if num_current_points > num_points:
                indices = torch.randperm(num_current_points)[:num_points]
                downsampled_points = downsampled_points[indices]
            elif num_current_points < num_points:
                padding = torch.zeros((num_points - num_current_points, lidar_point_cloud.size(1)), device=lidar_point_cloud.device)
                downsampled_points = torch.cat([downsampled_points, padding], dim=0)

        return downsampled_points.to('cpu')      
    
    def voxel_downsampling_open3d(self, points:np.array, voxel_size=0.1, num_points=50_000):
        # Convert NumPy array to Open3D PointCloud
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)

        # Apply voxelization
        voxelized_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        voxelized_points = np.asarray(voxelized_pcd.points)

        # Resample to ensure the desired number of points
        if num_points is not None:
            if len(voxelized_points) > num_points:
                indices = np.random.choice(len(voxelized_points), num_points, replace=False)
                voxelized_points = voxelized_points[indices]
            elif len(voxelized_points) < num_points:
                additional_indices = np.random.choice(len(voxelized_points), num_points - len(voxelized_points), replace=True)
                voxelized_points = np.vstack((voxelized_points, voxelized_points[additional_indices]))

        # Convert to PyTorch tensor
        torch_tensor = torch.tensor(voxelized_points, dtype=torch.float32)
        # final_num_points = torch_tensor.shape[0]
        return torch_tensor    
    
    def obtain_valid_lidar_points(self, projected_voxel_2d:torch.tensor, image_resize:tuple, 
                                  grid_sizes:Iterable[tuple]):
        
        valid_lidar_points_dict = {}
        
        W_img, H_img = image_resize        
        
        for grid_size in grid_sizes:
            W_grid, H_grid = grid_size
            # valid_lidar_points_dict[(W_grid, H_grid)] = {}
            
            grid_coords = projected_voxel_2d.clone()            
            grid_coords = torch.floor(grid_coords).long() # Discretize to grid cell indices
            
            # grid_coords = grid_coords[nonzero_depth_mask]
                        
            #Normalize to grid dimensions
            grid_coords[:, 0] = (grid_coords[:, 0] / W_img) * W_grid # X to grid width
            grid_coords[:, 1] = (grid_coords[:, 1] / H_img) * H_grid # Y to grid width                         
            
            valid_mask_x = (grid_coords[:, 0] >=0) & (grid_coords[:, 0] < W_grid)
            valid_mask_y = (grid_coords[:, 1] >=0) & (grid_coords[:, 1] < H_grid) 
            
            valid_grid_coords = grid_coords[valid_mask_x & valid_mask_y]                  

            # Map 2D grid coordinates to 1D indices for counting
            # Convert 2D grid indices (𝑥,𝑦) (x,y) to a 1D index using the formula: 
            # index = y * W_grid + x.
            linear_indices = valid_grid_coords[:, 1] * W_grid + valid_grid_coords[:, 0]

            # Count points per grid cell using scatter_add
            point_counts = torch.zeros(W_grid * H_grid, dtype=torch.int64)
            point_counts.scatter_add_(0, linear_indices, torch.ones_like(linear_indices, dtype=torch.int64))

            # Reshape to grid dimensions
            point_counts = point_counts.view(H_grid, W_grid)  
            # point_counts[point_counts  == 0] = 1e-5         
            
            valid_lidar_points_dict[grid_size] = {
                'valid_mask_x':valid_mask_x,
                'valid_mask_y':valid_mask_y,
                "valid_indices": valid_mask_x & valid_mask_y,
                "valid_grid_coords":valid_grid_coords,
                "count_grid":point_counts
            }
            
        return valid_lidar_points_dict    

class Kitti2DObjectDetectDataset(Dataset):
    
    def __init__(self, lidar_dir:str, 
                calibration_dir:str, 
                left_image_dir:str=None, 
                right_image_dir:str=None,
                labels_dir:str=None, 
                dataset_type:str="train"):
        
        self.left_image_dir = left_image_dir 
        self.right_image_dir = right_image_dir
                        
        if not bool(self.left_image_dir) and not bool(self.right_image_dir):
            raise Exception(f'Both Left and Right Images cannot be {self.left_image_dir}')
            
            
        self.calibration_dir = calibration_dir
        self.lidar_dir = lidar_dir
        self.labels_dir = labels_dir 
        self.dataset_type = dataset_type
        
        self.lidar_files = os.listdir(self.lidar_dir)
        self.left_image_files = os.listdir(self.left_image_dir)
        self.right_image_files = os.listdir(self.right_image_dir)
        self.calibration_files = os.listdir(self.calibration_dir)
        self.label_files = os.listdir(self.labels_dir)
        
    def __len__(self):
        return len(
            self.lidar_files
        )
        
    def __getitem__(self, idx):
        
        lidar_file = self.lidar_files[idx]
        _id = lidar_file.split('.')[0]
        
        return {
            'lidar_file_path':f'{self.lidar_dir}/{lidar_file}',
            'calibration_file_path':f'{self.calibration_dir}/{_id}.txt',
            'left_image_file_path': f'{self.left_image_dir}/{_id}.png' if self.left_image_dir is not None else None,
            'right_image_file_path':f'{self.right_image_dir}/{_id}.png' if self.right_image_dir is not None else None,
            'label_file_path':f'{self.labels_dir}/{_id}.txt' if self.labels_dir is not None else None
        }
        
class KittiLidarFusionCollateFn(object):
    
    def __init__(self, image_resize:list, original_size:tuple=(1242, 375),
                precomputed_voxel_dir:str=None, precomputed_proj2d_dir:str=None,
                transformation=None, clip_distance:float=2.0, apply_augmentation:bool=True, categorize_labels:bool=False,
                process_pointnet_inputs:bool=False, project_2d:bool=False, voxelization:bool=False):        
        
        self.image_resize = image_resize
        self.transformation = transformation
        self.clip_distance = clip_distance
        self.project_2d = project_2d
        self.voxelization = voxelization
        self.original_size = original_size
        self.process_pointnet_inputs = process_pointnet_inputs
        self.categorize_labels = categorize_labels
        
        self.voxelization_backend = "open3d"
        
        self.precomputed_voxel_dir = precomputed_voxel_dir if precomputed_voxel_dir is not None else ""
        self.precomputed_proj2d_dir = precomputed_proj2d_dir if precomputed_proj2d_dir is not None else ""
        
        self.grid_sizes = [(13, 13), (26, 26), (52, 52)]        
        
        if self.transformation is None:
            self.transformation = albumentations.Compose(                
                [albumentations.Resize(height=self.image_resize[0], width=self.image_resize[1], always_apply=True)],
                bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
            
            self.image_only_transformation = albumentations.Compose(
                [albumentations.Resize(height=self.image_resize[0], width=self.image_resize[1], always_apply=True)]
            )
            
        self.apply_augmentation = apply_augmentation
        self.image_augmentor = albumentations.Compose([
            albumentations.OneOf([
                albumentations.GaussianBlur(blur_limit=(1, 3), p=0.5),
                albumentations.MedianBlur(blur_limit=3, p=0.5),
                albumentations.MotionBlur(blur_limit=3, p=0.5)
            ], p=0.5),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                albumentations.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            ], p=0.5),
            albumentations.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
            albumentations.OneOf([
                albumentations.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, fill_value=0, p=0.5),
                albumentations.CropAndPad(percent=(-0.1, 0.1), pad_mode=cv2.BORDER_CONSTANT, p=0.5),
            ], p=0.5)
        ])                    
        
    def read_calibration_file(self, calib_file_path):
        calibration_dict = {}
        
        with open(calib_file_path, 'r') as f:
            for line in f.readlines():
                if line != '\n':
                    key, value = line.split(':')
                    calibration_dict[key.strip()] = np.fromstring(
                        value, sep=' '
                    )
                    
        return calibration_dict
    
    def categorize_label_difficulty(self, height, truncation, occlusion):
        # if height >= 40.0 and occlusion == 0.0 and truncation <= 0.15:
        #     # return 'easy'
        #     return 0
        # elif height >= 25.0 and occlusion <= 1.0 and truncation <= 0.30:
        #     # return 'moderate'
        #     return 1
        # elif height >= 25.0 and occlusion <= 2.0 and truncation <= 0.50:
        #     # return 'hard'
        #     return 2
        
        # Easy: Fully visible, truncation <= 15%
        if occlusion == 0 and truncation <= 0.15:
            return 0  # Easy

        # Moderate: Partly occluded, truncation <= 30%
        elif occlusion <= 1 and truncation <= 0.30:
            return 1  # Moderate

        # Hard: Difficult to see, truncation <= 50%
        elif occlusion <= 2 and truncation <= 0.50:
            return 2  # Hard
        
        return 2
        
    def read_label_file(self, label_file_path:str):
        '''
        #Values    Index    Name      Description
        ----------------------------------------------------------------------------
        1        0       type      Describes the type of object: 'Car', 'Van', 'Truck',
                                    'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                    'Misc' or 'DontCare'
        1        1       truncated Float from 0 (non-truncated) to 1 (truncated), where
                                    truncated refers to the object leaving image boundaries
        1        2       occluded  Integer (0,1,2,3) indicating occlusion state:
                                    0 = fully visible, 1 = partly occluded
                                    2 = largely occluded, 3 = unknown
        1        3       alpha     Observation angle of object, ranging [-pi..pi]
        4        4-7       bbox      2D bounding box of object in the image (0-based index):
                                    contains left, top, right, bottom pixel coordinates
        3        8-10       dimensions 3D object dimensions: height, width, length (in meters)
        3        11-13       location  3D object location x,y,z in camera coordinates (in meters)
        1        14       rotation_y Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1        15       score     Only for results: Float, indicating confidence in
                                    detection, needed for p/r curves, higher is better.

        '''
        class_labels = []
        bboxes = []
        categories = []
        
        # categories = {'easy': [], 'moderate': [], 'hard': []}
        
        with open(label_file_path, 'r') as file:
            for line in file:                
                parts = line.strip().split() # Split the line into parts

                # Extract class label and bounding box coordinates
                obj_type = parts[0]  # First element is the object type
                
                if Enums.mapping_dict and obj_type in Enums.mapping_dict:
                    obj_type = Enums.mapping_dict[obj_type]
                
                elif obj_type not in Enums.KiTTi_label2Id:
                    continue  # Skip invalid object types

                # Bounding box coordinates
                left = float(parts[4])  # left
                top = float(parts[5])  # top
                right = float(parts[6])  # right
                bottom = float(parts[7])  # bottom

                class_id = Enums.KiTTi_label2Id[obj_type]
                
                class_labels.append(class_id)
                bboxes.append([left, top, right, bottom])
                
                if self.categorize_labels:
                    difficulty = self.categorize_label_difficulty(
                        float(parts[8]), float(parts[1]), int(parts[2])
                    )
                    
                    # categories[difficulty] = (class_id, [left, top, right, bottom])
                    categories.append(difficulty)
                
        if self.categorize_labels:
            return class_labels, bboxes, categories
        else:
            return class_labels, bboxes

    def transform_sample(self, image:np.array, label_bboxes:np.array=None, class_labels:np.array=None):                        
        
        if label_bboxes is not None and class_labels is not None:        
            transformed_dict = self.transformation(
                image=image, bboxes=label_bboxes, class_labels=class_labels
            )
        
        else:
            transformed_dict = self.image_only_transformation(
                image=image
            )            
        
        return transformed_dict
    
    def prepare_targets(self, batch_idx:int, class_labels:list, class_bboxes:list):

        targets = []
        for bbox, class_id in zip(class_bboxes, class_labels):        
            left, top, right, bottom = bbox

            x_center = (left + right) / 2 / self.image_resize[1]
            y_center = (top + bottom) / 2 / self.image_resize[0]
            width = (right - left) / self.image_resize[1]
            height = (bottom - top) / self.image_resize[0]

            targets.append([
                batch_idx, class_id, x_center, y_center, width, height
            ])
            
        return targets    

    def create_maps(self, projected_points:np.array, y_max:int, x_max:int, intensities:np.array, depths:np.array, heights:np.array):

        reflectance_map = np.zeros((y_max, x_max), dtype=np.float32)
        depth_map = np.zeros((y_max, x_max), dtype=np.float32)       
        
        for i in range(len(projected_points)):
            x, y = int(projected_points[i, 0]), int(projected_points[i, 1])
            if 0 <= x < x_max and 0 <= y < y_max:
                # Use maximum intensity for reflectance
                reflectance_map[y, x] = max(reflectance_map[y, x], intensities[i])
                # Use the closest depth value
                if depth_map[y, x] == 0:  # if depth is not set yet
                    depth_map[y, x] = depths[i]        

        reflectance_map = (reflectance_map / np.max(reflectance_map) * 255).astype(np.uint8)
        depth_map = (depth_map / np.max(depth_map) * 255).astype(np.uint8)
        
        return depth_map, reflectance_map

    def preprocess(self, lidar_point_cloud:np.array, calibration_dict:dict, image_array:np.array):
        
        points_2d, depths,_ = LidarPreprocessorUtils().transform_lidar_points(
            lidar_point_cloud, calibration_dict
        )        
        
        x_min, y_min = 0, 0
        x_max, y_max = image_array.shape[1], image_array.shape[0]        
        
        fov_inds = (
                (points_2d[:, 0] < x_max)
                & (points_2d[:, 0] >= x_min)
                & (points_2d[:, 1] < y_max)
                & (points_2d[:, 1] >= y_min)
        )    
        
        fov_inds = fov_inds & (
                    lidar_point_cloud[:, 0] > self.clip_distance)      
        
        projected_points = points_2d[fov_inds]
        heights = lidar_point_cloud[fov_inds, 2]
        
        depth_map, reflectance_map = self.create_maps(
            projected_points, y_max, x_max, 
            lidar_point_cloud[fov_inds, -1],
            lidar_point_cloud[fov_inds, -2],
            heights            
        )         
        
        return depth_map, reflectance_map, projected_points
    
    def preprocess_yolo_inputs(self, lidar_point_cloud:np.array, calibration_dict:dict, 
                               left_image_arr=None, right_image_arr=None):
        if left_image_arr is not None:
            depth_map, reflectance_map, _ = self.preprocess(lidar_point_cloud, calibration_dict, left_image_arr)
            depth_map, reflectance_map = np.expand_dims(depth_map, axis=-1), np.expand_dims(reflectance_map, axis=-1)            
            
            combined_image = np.concatenate([
                left_image_arr, depth_map, reflectance_map
            ], axis=-1)

        elif right_image_arr is not None:
            depth_map, reflectance_map, _ = self.preprocess(lidar_point_cloud, calibration_dict, right_image_arr)  
            depth_map, reflectance_map = np.expand_dims(depth_map, axis=-1), np.expand_dims(reflectance_map, axis=-1)
            
            combined_image = np.concatenate([
                right_image_arr, depth_map, reflectance_map
            ], axis=-1)                  
            
        if self.apply_augmentation:
            combined_image = self.image_augmentor(image=combined_image)['image']                
                
        return combined_image

    def preprocess_pointnet_inputs(self, lidar_point_cloud:np.array, lidar_file_path:str,
                                   calibration_dict:dict):

        lidar_fn = lidar_file_path.split('/')[-1]
        _id = lidar_fn.split('.')[0]           
        voxel_file_path = f'{self.precomputed_voxel_dir}/{_id}_voxelized.bin'
                                                
        if os.path.exists(voxel_file_path):                        
            voxelized_point_cloud = np.fromfile(voxel_file_path,
                                                dtype=np.float32).reshape(-1, 3)
            voxelized_point_cloud = torch.from_numpy(voxelized_point_cloud)
            voxelized_point_cloud = voxelized_point_cloud.transpose(1, 0)

        else:
            voxelized_point_cloud = torch.tensor(lidar_point_cloud)   
            if self.voxelization_backend == "torch":
                voxelized_point_cloud = LidarPreprocessorUtils().voxel_downsampling_torch(torch.from_numpy(lidar_point_cloud), 
                                                                    voxel_size=0.2,
                                                                    num_points=50_000)
            if self.voxelization_backend == "open3d":
                voxelized_point_cloud = LidarPreprocessorUtils().voxel_downsampling_open3d(lidar_point_cloud[:, :3], 
                                                                    voxel_size=0.2,
                                                                    num_points=50_000)
            
            voxelized_point_cloud = voxelized_point_cloud[:, :3].transpose(1, 0)            

        valid_lidar_points_dict = {}
        
        if self.project_2d:
            if os.path.exists(f'{self.precomputed_proj2d_dir}/{_id}'):
                fn = f'{self.precomputed_proj2d_dir}/{_id}'
                
                for grid_size in self.grid_sizes:
                    valid_indices = np.fromfile(
                        f'{fn}/{grid_size}_valid_indices.bin',
                        dtype=np.int8
                    ).astype(bool)                            
                    
                    valid_grid_coords = np.fromfile(
                        f'{fn}/{grid_size}_valid_grid_coords.bin',
                        dtype=np.int64
                    ).astype(np.int64).reshape(-1, 2)                                                      

                    grid_coords = np.fromfile(
                        f'{fn}/{grid_size}_grid_coords.bin',
                        dtype=np.int64
                    ).astype(np.int64).reshape(-1, 2)                                                      

                    valid_lidar_points_dict[grid_size] = {
                        'valid_indices':torch.from_numpy(valid_indices),
                        "valid_grid_coords":torch.from_numpy(valid_grid_coords),
                        "grid_coords":torch.from_numpy(grid_coords)
                    }
                    
            else:
                projected_voxel_2d, depth, negative_depth_mask = LidarPreprocessorUtils().transform_lidar_points(
                    voxelized_point_cloud.transpose(0, 1).cpu().numpy(), calibration_dict
                )
                
                # depth[negative_depth_mask] = 1e-5
                # projected_voxel_2d[:, 0] /= depth
                # projected_voxel_2d[:, 1] /= depth
                
                projected_voxel_2d[:, 0] = (projected_voxel_2d[:, 0]/ self.original_size[0]) * (self.image_resize[0])
                projected_voxel_2d[:, 1] = (projected_voxel_2d[:, 1]/ self.original_size[1]) * (self.image_resize[1])                            
                                
                projected_voxel_2d = torch.from_numpy(projected_voxel_2d)
                valid_lidar_points_dict = LidarPreprocessorUtils().obtain_valid_lidar_points(
                    projected_voxel_2d, self.image_resize, self.grid_sizes
                )
                
        return valid_lidar_points_dict, voxelized_point_cloud       

    def __call__(self, batch_data_filepaths:List[Dict]):

        batch_data_items = {
            "images": [], #list of tensors --> stack --> tensor (bs, n_c, h, w),
            "targets": [],
            "image_paths":[],
            "raw_point_clouds":[],
            "proj2d_pc_mask":[], 
            'target_difficulty':[]
            # "bboxes":[], # list of list of list [bs * [num_labels * [x,y,x,y] ]] (inner list of 4 elements)
            # "class_labels":[] # list of list of class_labels [bs * [num_labels] ] (inner list is class_ids)
        }        
        
        for idx, file_path_dict in enumerate(batch_data_filepaths):
            lidar_file_path = file_path_dict['lidar_file_path']
            calibration_file_path = file_path_dict['calibration_file_path']
            left_image_file_path = file_path_dict['left_image_file_path']
            right_image_file_path = file_path_dict['right_image_file_path']
            label_file_path = file_path_dict['label_file_path']            
            
            if os.path.exists(calibration_file_path):
                calibration_dict = self.read_calibration_file(calibration_file_path)
            else:
                print(f'Calib {calibration_file_path} Does not Exist!')
                
            left_image_arr = None
            right_image_arr = None                

            if bool(left_image_file_path) and os.path.exists(left_image_file_path):
                left_image_arr = cv2.imread(left_image_file_path)
                batch_data_items['image_paths'].append(left_image_file_path)
            
            if bool(right_image_file_path) and os.path.exists(right_image_file_path):
                right_image_arr = cv2.imread(right_image_file_path)    
                batch_data_items['image_paths'].append(right_image_file_path)
                
            if left_image_arr is None and right_image_arr is None:
                print(f'Left Image Path {left_image_file_path} and Right Image Path {right_image_file_path}')
                exit(1)
                
            if os.path.exists(lidar_file_path):
                lidar_point_cloud = np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4)
                
                combined_image = self.preprocess_yolo_inputs(
                    lidar_point_cloud, calibration_dict, 
                    left_image_arr, right_image_arr
                )   
                combined_image = combined_image.astype('float32')  
                combined_image /= 255.
                
                if self.process_pointnet_inputs:                
                    valid_lidar_points_dict, voxelized_point_cloud = self.preprocess_pointnet_inputs(
                        lidar_point_cloud, lidar_file_path, calibration_dict
                    )                
                    
                    batch_data_items['proj2d_pc_mask'].append(valid_lidar_points_dict) 
                    batch_data_items['raw_point_clouds'].append(voxelized_point_cloud)                  
                
                if label_file_path is not None:
                    if os.path.exists(label_file_path):
                        if self.categorize_labels:
                            class_labels, label_bboxes, target_difficulty = self.read_label_file(label_file_path) 

                            batch_data_items['target_difficulty'].append(
                                torch.tensor(target_difficulty, dtype=torch.uint8)
                            )
                            

                        else:
                            class_labels, label_bboxes = self.read_label_file(label_file_path)
                        
                        # print(label_bboxes)
                        
                        transformed_dict = self.transform_sample(
                            combined_image, label_bboxes, class_labels
                        )      
                        
                        # print(transformed_dict['bboxes'])
                        # exit(1)
                        
                        targets = self.prepare_targets(
                            idx, transformed_dict['class_labels'], transformed_dict['bboxes']
                        )                                                                              
                        
                        batch_data_items['targets'].append(
                            torch.tensor(targets, dtype=torch.float32)
                        )
                    else:
                        print(f'Label File Path not found!!')
                        exit(1)                    
                        
                else:
                    transformed_dict = self.transform_sample(
                        combined_image
                    )                                                

                image_tensor = torch.from_numpy(transformed_dict['image']).permute((2, 0, 1))            
                batch_data_items['images'].append(image_tensor)              

                
            else:
                ''' 
                TODO, add RGB only inputs preprocessing here. 
                '''
                print(f'Lidar {lidar_file_path} Does not Exist!')
                exit(1)            
                
        batch_data_items['images'] = torch.stack(
            batch_data_items['images'], dim=0
        ).float()
        
        if batch_data_items['raw_point_clouds']:        
            batch_data_items['raw_point_clouds'] = torch.stack(
                batch_data_items['raw_point_clouds'], dim=0
            )                        
        
        if batch_data_items['targets']:
            batch_data_items['targets'] = torch.concat(
                batch_data_items['targets'], dim=0
            )
            
        if batch_data_items['target_difficulty']:
            batch_data_items['target_difficulty'] = torch.concat(
                batch_data_items['target_difficulty'], dim=0
            )
            
        return batch_data_items
        