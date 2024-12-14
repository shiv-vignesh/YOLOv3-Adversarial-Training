from __future__ import division

import os
from itertools import chain
from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_utils import parse_model_config


def create_modules(module_defs: List[dict]) -> Tuple[dict, nn.ModuleList]:
    """
    Constructs module list of layer blocks from module configuration in module_defs

    :param module_defs: List of dictionaries with module definitions
    :return: Hyperparameters and pytorch module list
    """
    hyperparams = module_defs.pop(0)
    hyperparams.update({
        'batch': int(hyperparams['batch']),
        'subdivisions': int(hyperparams['subdivisions']),
        'width': int(hyperparams['width']),
        'height': int(hyperparams['height']),
        'channels': int(hyperparams['channels']),
        'optimizer': hyperparams.get('optimizer'),
        'momentum': float(hyperparams['momentum']),
        'decay': float(hyperparams['decay']),
        'learning_rate': float(hyperparams['learning_rate']),
        'burn_in': int(hyperparams['burn_in']),
        'max_batches': int(hyperparams['max_batches']),
        'policy': hyperparams['policy'],
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),
                             map(float, hyperparams["scales"].split(","))))
    })
    
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    output_filters = [hyperparams["channels"]]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif module_def["activation"] == "mish":
                modules.add_module(f"mish_{module_i}", nn.Mish())
            elif module_def["activation"] == "logistic":
                modules.add_module(f"sigmoid_{module_i}", nn.Sigmoid())
            elif module_def["activation"] == "swish":
                modules.add_module(f"swish_{module_i}", nn.SiLU())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers]) // int(module_def.get("groups", 1))
            modules.add_module(f"route_{module_i}", nn.Sequential())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            new_coords = bool(module_def.get("new_coords", False))
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, new_coords)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode: str = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors: List[Tuple[int, int]], num_classes: int, new_coords: bool):
        """
        Create a YOLO layer

        :param anchors: List of anchors
        :param num_classes: Number of classes
        :param new_coords: Whether to use the new coordinate format from YOLO V7
        """
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.new_coords = new_coords
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor, 5 --> xywh & objectness_score
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

    def forward(self, x: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Forward pass of the YOLO layer

        :param x: Input tensor
        :param img_size: Size of the input image
        """
        stride = img_size // x.size(2)
        self.stride = stride
        
        # x(bs,(num_classes + 5) * num_anchors,x,y) to x(bs,3,x,y, num_classes + 5)
        # x, y --> [(13,13),(26,26),(52,52)]
        bs, _, ny, nx = x.shape  
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if not self.training:
            x = x.view(bs, -1, self.no)
        
        return x

        # if not self.training:  # inference
        #     # print(f'Within Inference If')
        #     if self.grid.shape[2:4] != x.shape[2:4]:
        #         self.grid = self._make_grid(nx, ny).to(x.device)
                

            # if self.new_coords:
            #     x[..., 0:2] = (x[..., 0:2] + self.grid) * stride  # xy
            #     x[..., 2:4] = x[..., 2:4] ** 2 * (4 * self.anchor_grid) # wh     
            #     print(f'New Coords')           
            # else:
            #     # x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
            #     # x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
            #     # x[..., 4:] = x[..., 4:].sigmoid() # conf, cls
                
            #     '''Commented sigmoid() as it results in NaN while computing loss'''
            #     '''Applying sigmoid within trainer while computing IOU and Accuracy'''
                
            #     # x[..., 0:2] = (x[..., 0:2] + self.grid) * stride  # xy
            #     # x[..., 2:4] = x[..., 2:4] ** 2 * (4 * self.anchor_grid) # wh 
                
            #     print(f'Last Else')
                              
            
            # exit(1)


    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        """
        Create a grid of (x, y) coordinates

        :param nx: Number of x coordinates
        :param ny: Number of y coordinates
        """
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        
        self.identify_detection_head_indices()

    def identify_detection_head_indices(self):                
        
        ''' 
        REQUIRED For Detection head forward pass after adaptive fusion.
        contains tuple of [(i-1, i)] or [(i, i+1)]
            i-1 or i:
                preceeding conv that downsamples from 
                latent_space to filters=$(expr 3 \* $(expr $NUM_CLASSES \+ 5))
            i or i+1: 
                YOLO detection head 
        '''
        self.detection_head_indices = []
        self.detection_head_names = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if (
                module_def["type"] == "convolutional"
                and i + 1 < len(self.module_defs)
                and self.module_defs[i + 1]["type"] == "yolo"
            ):
                self.detection_head_indices.append(
                    (i, i+1)
                )               
                self.detection_head_names.append(
                    (self.module_defs[i], self.module_defs[i+1])
                )         

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:            
                x = module(x)                
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        # return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)        
        return yolo_outputs
    
    def forward_backbone(self, x):
        img_size = x.size(2)
        layer_outputs = []
        
        intermediate_features = []
        
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:            
                if (
                    module_def["type"] == "convolutional"
                    and i + 1 < len(self.module_defs)
                    and self.module_defs[i + 1]["type"] == "yolo"
                ):
                    intermediate_features.append(x)
                
                x = module(x)                                
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            layer_outputs.append(x)
    
        return intermediate_features
    
    def forward_detection_head(self, grid_features:Iterable[torch.tensor], image_size:int):
        
        assert len(grid_features) == len(self.detection_head_indices)
        
        for idx, (prev_conv_idx, det_head_idx) in enumerate(self.detection_head_indices):
            
            grid_features[idx] = self.module_list[prev_conv_idx](grid_features[idx])
            grid_features[idx] = self.module_list[det_head_idx][0](grid_features[idx], image_size)
            
        return grid_features            
    
    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # Conv2d(5, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                # Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                
                ''' 
                Original input layer is 
                    torch.Size([32, 3, 3, 3])
                Modified input layer is 
                    torch.Size([32, 3 + lidar_channels, 3, 3])
                    
                Numel() from original = 864
                '''
                                
                if i == 0 and self.hyperparams['channels'] > 3:                    
                    numel_original = 864                                        
                    original_weights = torch.from_numpy(
                        weights[ptr:ptr+numel_original]).view(
                            [32, 3, 3, 3]
                        )
                    
                    repeating_channels = self.hyperparams['channels'] - 3
                    avg_weights = original_weights.mean(dim=1, keepdim=True)   
                    avg_weights = avg_weights.repeat(1, repeating_channels, 1, 1)                 
                    
                    new_weights = torch.cat((original_weights, avg_weights), dim=1)
                    conv_layer.weight.data.copy_(new_weights)                    

                    ptr += numel_original
                    continue
                
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def get_backbone_trainable_params(self, requires_grad:bool):
        
        indices = [idx for tup in self.detection_head_indices for idx in tup]
        backbone_params = []
        
        for i, module in enumerate(self.module_list):
            if i not in indices:
                for name, p in module.named_parameters():
                    backbone_params.append(p)
                    p.requires_grad = requires_grad
                    
        return backbone_params
        
    def get_detection_head_params(self, requires_grad:bool):
        
        detection_head_params = []
        
        for idx_pair in self.detection_head_indices:
            for module_idx in idx_pair:
                for p in self.module_list[module_idx].parameters():
                    p.requires_grad = requires_grad
                    detection_head_params.append(p)    
                    
        return detection_head_params       
