import os, cv2
import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import numpy as np
import random

from terminaltables import AsciiTable

from model.yolo import Darknet
from model.yolo_utils import xywh2xyxy, reshape_outputs, apply_sigmoid_activation, non_max_suppression, rescale_boxes, get_batch_statistics, ap_per_class
from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from dataset_utils.enums import Enums

from trainer.trainer_adversarial import AdversarialAttack

def print_eval_stats(metrics_output, class_names, verbose=True):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP", "precision", "recall", "F1"]]
            for i, c in enumerate(ap_class):
                # ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                ap_table.append([
                    c, 
                    class_names[c],
                    f'{AP[i]:.5f}',
                    f'{precision[i]:.5f}',
                    f'{recall[i]:.5f}',
                    f'{f1[i]:.5f}'
                ])
            
            table_string = AsciiTable(ap_table).table
            
            print(f'---------- mAP per Class----------')
            print(f'{table_string}')
        
            print(f'---------- Total mAP {AP.mean():.5f} ----------')
            
    else:
        print("---- mAP not measured (no detections found by model) ----")                   


def load_model(config_path:str, ckpt_path:str):
    device = torch.device("cuda" if torch.cuda.is_available()
                        else "cpu")  # Select device for inference    
    model = Darknet(config_path)
    
    if os.path.exists(f'{ckpt_path}'):
        if ckpt_path.endswith('.pth'):
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device)
            )            
        else:
            model.load_darknet_weights(ckpt_path)
        
        print(f'Successful!! loaded weight from {ckpt_path} --- Device: {device}')
    
    else:
        print(f'Unable to locate {ckpt_path}')
        exit(1)
        
    model.to(device)

    return model, device
    
def create_dataloader(test_dataset_kwargs:dict):
    dataset = Kitti2DObjectDetectDataset(
        lidar_dir=test_dataset_kwargs['lidar_dir'],
        calibration_dir=test_dataset_kwargs['calibration_dir'],
        left_image_dir=test_dataset_kwargs['left_image_dir'],
        right_image_dir=test_dataset_kwargs['right_image_dir'],
        labels_dir=test_dataset_kwargs['labels_dir']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=test_dataset_kwargs['batch_size'],
        collate_fn=KittiLidarFusionCollateFn(
            image_resize=test_dataset_kwargs['image_resize']
        ),
        shuffle=test_dataset_kwargs['shuffle']
    )
    
    return dataloader

def draw_and_save_output_images(image_detections:list, 
                                image_paths:list, resized_image_size:int,
                                output_path:str, classes:list):
    
    for (image_path, detections) in zip(image_paths, image_detections):
        image_arr = cv2.imread(image_path)
        image_arr = image_arr[:, :, ::-1]
        
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(image_arr)      
        
        detections = rescale_boxes(detections, resized_image_size, image_arr.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
        bbox_colors = random.sample(colors, n_cls_preds)      
        
        for x1, y1, x2, y2, conf, cls_pred in detections:

            # print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=f"{classes[int(cls_pred)]}: {conf:.2f}",
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0})

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = os.path.basename(image_path).split(".")[0]
        filepath = os.path.join(output_path, f"{filename}.png")
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0.0)
        plt.close()                  
        

def test(config_path:str, ckpt_path:str, test_dataset_kwargs:dict, output_dir:str):    
    
    model, device = load_model(config_path, ckpt_path)    
    test_dataloader = create_dataloader(test_dataset_kwargs)
    
    if not os.path.exists(f'{output_dir}'):
        os.makedirs(output_dir)
        
    model.eval()
    
    image_paths = []
    image_detections = []
    
    test_iter = tqdm(test_dataloader)
    
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    img_size = model.hyperparams['height']
    
    attack = AdversarialAttack(model, attack_type='fgsm')
    
    for batch_idx, data_items in enumerate(test_iter):
        for k,v in data_items.items():
            if torch.is_tensor(v):                    
                data_items[k] = v.to(device)   
                
        perturbed_images = attack(data_items['images'], data_items['targets'], attack_channels=[3, 4])
                
        targets = data_items['targets'].cpu()
        labels += targets[:, 1] #[class_id] 
                    
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size                
                
        with torch.no_grad():
            outputs = model(perturbed_images)        
            anchor_grids = [yolo_layer.anchor_grid for yolo_layer in model.yolo_layers]            
            outputs = reshape_outputs(outputs)            
            outputs = apply_sigmoid_activation(outputs, perturbed_images.size(2), anchor_grids)                
            outputs = non_max_suppression(outputs)       
            
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.45)     
            
        image_detections.extend(outputs)
        image_paths.extend(data_items['image_paths'])
        
        if (batch_idx + 1) % 50 == 0:
                
            class_names = list(Enums.KiTTi_label2Id.keys())         
            draw_and_save_output_images(
                image_detections, image_paths, test_dataset_kwargs['image_resize'][0],
                output_dir, class_names
            )
            
            image_detections = []
            image_paths = []
        
        # if batch_idx > 10:
        #     break            

    if image_detections:
        class_names = list(Enums.KiTTi_label2Id.keys())  
        draw_and_save_output_images(
            image_detections, image_paths, test_dataset_kwargs['image_resize'][0],
            output_dir, class_names
        )
        
        image_detections = []
        image_paths = []        
        
    print(f'Detection Finished! Computing Metrics')
    
        # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]            

    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels) 
    
    class_names = list(Enums.KiTTi_label2Id.keys())    
    print_eval_stats(metrics_output, class_names)

if __name__ == "__main__":
    
    ''' 
    TODO 
    1. Set fixed colors for each class bbox
    2. Distribution of TP and FP vs confidence score. 
    3. 
    '''
    
    test_kwargs = {
        "config_path":"config/yolov3-yolo_reduced_classes.cfg",
        "ckpt_path":"adversarial_training_kitti_pgd/yolo_weights_2.pth",
        "test_dataset_kwargs":{
            "lidar_dir":"../CSCI739/data/velodyne/validation/velodyne",
            "calibration_dir":"../CSCI739/data/calibration/validation/calib",
            "left_image_dir":"../CSCI739/data/left_images/validation/image_2",
            "right_image_dir":None,
            "labels_dir":"../CSCI739/data/labels/validation/label_2",
            "batch_size":16,
            "shuffle":False, 
            "image_resize":[416, 416]
        },
        "output_dir":"adversarial/all_channels_attack"
    }
    
    test(**test_kwargs)