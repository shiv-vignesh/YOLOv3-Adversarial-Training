from tqdm import tqdm
import torch, time
import os, math, copy
from typing import Iterable
import numpy as np
from torch.utils.data import DataLoader
from terminaltables import AsciiTable

from .logger import Logger
from model.yolo import Darknet
from model.yolo_utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from dataset_utils.enums import Enums
from trainer.loss import compute_loss

from trainer.trainer_adversarial import AdversarialAttack, AdversarialTrainer

class FeatureAlignmentTrainer(AdversarialTrainer):
    def __init__(self, model:Darknet, 
                dataset_kwargs:dict, optimizer_kwargs:dict,
                trainer_kwargs:dict, lr_scheduler_kwargs:dict):
        
        super(FeatureAlignmentTrainer, self).__init__(model, dataset_kwargs, optimizer_kwargs, trainer_kwargs, lr_scheduler_kwargs)
                
        if trainer_kwargs['alignment_type'] == "feature_alignment":
            self.alignment_type = trainer_kwargs['alignment_type'] 
            self.clean_model = copy.deepcopy(self.model)
            
            for param in self.clean_model.parameters():
                param.requires_grad_ = False
            
        elif trainer_kwargs['alignment_type'] == "irl":  
            self.alignment_type = trainer_kwargs['alignment_type'] 
            self.clean_model = None
            
    def compute_feature_distances(self, clean_features:Iterable[torch.tensor], perturbed_feautres:Iterable[torch.tensor]):
        
        feature_distances = torch.zeros(len(clean_features), dtype=torch.float32, device=self.device)
        
        for idx, (clean_feature, perturbed_feature) in enumerate(zip(clean_features, perturbed_feautres)):
            l2 = torch.norm(clean_feature - perturbed_feature, p=2, dim=(1, 2, 3)).mean(dim=0)
            feature_distances[idx] += l2 * 0.01
            
        return torch.mean(feature_distances, dim=0)
    
    def train_one_epoch(self):
        
        self.model.train()
        
        total_loss = 0.0 
        ten_percent_batch_total_loss = 0
        
        total_clean_loss, total_perturbed_loss = 0.0, 0.0
        total_feature_distance = 0.0
        
        ten_percent_batch_clean_loss, ten_percent_batch_perturbed_loss = 0.0, 0.0
        ten_percent_batch_feature_distance = 0.0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0        
        
        train_iter = tqdm(self.train_dataloader)               
        
        for batch_idx, data_items in enumerate(train_iter):
            for k,v in data_items.items():
                if torch.is_tensor(v):                    
                    data_items[k] = v.to(self.device)
            
            step_begin_time = time.time()
            loss_dict = self.train_one_step(data_items)
            step_end_time = time.time()      
            
            clean_loss, perturbed_loss = loss_dict['clean_loss'], loss_dict['perturbed_loss']
            feature_distance = loss_dict['feature_distance']
    
            batches_done = len(self.train_dataloader) * self.cur_epoch + batch_idx   
            if batches_done % self.model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = self.model.hyperparams['learning_rate']
                if batches_done < self.model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / self.model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in self.model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                # self.logger.log_message(f"train/learning_rate, - lr {lr} - batches_done {batches_done}")
    
                # Set learning rate
                for g in self.optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                self.optimizer.step()
                # Reset gradients
                self.optimizer.zero_grad()      
                
            total_loss += clean_loss + perturbed_loss
            ten_percent_batch_total_loss += clean_loss + perturbed_loss
            
            total_clean_loss += clean_loss
            ten_percent_batch_clean_loss += clean_loss
            
            total_perturbed_loss += perturbed_loss
            ten_percent_batch_perturbed_loss += perturbed_loss
            
            total_feature_distance += feature_distance
            ten_percent_batch_feature_distance += feature_distance
            
            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)            

            if (batch_idx + 1) % self.ten_percent_train_batch == 0:
                average_loss = ten_percent_batch_total_loss/self.ten_percent_train_batch
                average_time = ten_percent_training_time/self.ten_percent_train_batch    
                
                average_clean_loss = ten_percent_batch_clean_loss/self.ten_percent_train_batch
                average_perturbed_loss = ten_percent_batch_perturbed_loss/self.ten_percent_train_batch
                avg_feature_distance = ten_percent_batch_feature_distance/self.ten_percent_train_batch
                
                message = f'Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - clean loss {average_clean_loss:.4f} - perturbed loss {average_perturbed_loss:.4f} - feature distance {avg_feature_distance:.4f} - total loss {average_loss:.4f}'
                self.logger.log_message(message=message)
                
                ten_percent_batch_total_loss = 0
                ten_percent_training_time = 0.0         
                
                ten_percent_batch_clean_loss, ten_percent_batch_perturbed_loss = 0.0, 0.0    
                ten_percent_batch_feature_distance = 0.0
                
        total_loss = total_loss/self.total_train_batch 
        total_clean_loss = total_clean_loss/self.total_train_batch
        total_perturbed_loss = total_perturbed_loss/self.total_train_batch
        total_feature_distance = total_feature_distance/self.total_train_batch
        
        self.logger.log_message(
            f'Epoch {self.cur_epoch} - Average Clean Loss {total_clean_loss:.4f} - Average Perturbed Loss {total_perturbed_loss:.4f} - feature distance {total_feature_distance:.4f} - Average Total Loss {total_loss:.4f}'
        )                              

    
    def train_one_step(self, data_items):            
        perturbed_images = self.adv_attack(data_items['images'], data_items['targets'])            
        if self.alignment_type == "irl":        
            with torch.set_grad_enabled(True):          
                clean_features = self.model.forward_backbone(data_items['images'])                                        
                perturbed_features = self.model.forward_backbone(perturbed_images)
                
                feature_distance = self.compute_feature_distances(clean_features, perturbed_features)
                feature_distance.backward(retain_graph=True)                
                                    
                perturbed_predictions = self.model.forward_detection_head(perturbed_features, perturbed_images.shape[2])
                clean_predictions = self.model.forward_detection_head(clean_features, data_items['images'].shape[2])
                
                clean_loss, clean_loss_components = compute_loss(clean_predictions, data_items['targets'], self.model)   
                clean_loss.backward(retain_graph=True)                
                
                perturbed_loss, perturbed_loss_components = compute_loss(perturbed_predictions, data_items['targets'], self.model)                    
                perturbed_loss.backward()
                

            return {
                'clean_loss':clean_loss.item(),
                'perturbed_loss':perturbed_loss.item(),
                'feature_distance':feature_distance.item()
            }
                    
                                                

                    
