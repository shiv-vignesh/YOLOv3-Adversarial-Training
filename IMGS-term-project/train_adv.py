import os, json
import torch

from model.yolo_utils import weights_init_normal
from model.yolo import Darknet

from trainer.trainer_adversarial import AdversarialTrainer

def load_torch_mismatch_weights(model:Darknet, pth_file:str):
    
    weights_dict = torch.load(pth_file, map_location='cpu')        
    model_dict = model.state_dict()
    
    for idx, (model_layer, weight_layer) in enumerate(zip(model_dict, weights_dict)):

        layer_shape = model_dict[model_layer].shape
        weight_shape = weights_dict[weight_layer].shape
        
        if layer_shape == weight_shape:
            model_dict[model_layer] = weights_dict[weight_layer]
            
        else:
            print(f'Mismatch at {model_layer}: Weights Shape: {weight_shape} - Layer Shape: {layer_shape}')
            if 'weight' in model_layer:
                model_dict[model_layer] = torch.nn.init.normal_(
                    model_dict[model_layer], 0.0, 0.02
                )
                
            elif 'bias' in model_layer:
                model_dict[model_layer] = torch.nn.init.constant_(
                    model_dict[model_layer], 0.0
                )                
            
    return model.load_state_dict(model_dict)

def load_from_darknet53(config_path:str, weights_path:str, ):
    
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    
    model = Darknet(config_path).to(device)
    model.apply(weights_init_normal)
    
    model.load_darknet_weights(weights_path)    

    return model  

def load_model(config_path:str, weights_path:str=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    
    model = Darknet(config_path).to(device)
    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            try:
                model.load_state_dict(torch.load(weights_path, map_location=device))
            except:
                print(f'Loading with Mismatch')
                load_torch_mismatch_weights(model.cpu(), weights_path)
                print(f'Done Loading')  
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model

if __name__ == "__main__":
    yolo_weights_pth_path = "adversarial_training_kitti/yolo_weights_29.pth"
    darknet53_path = ''
    cfg_file_path = 'config/yolov3-yolo_reduced_classes.cfg'
    
    trainer_config = json.load(open('config/yolo_adv_trainer.json'))
    
    if os.path.exists(darknet53_path):
        model = load_from_darknet53(cfg_file_path,darknet53_path)
    
    else:
        if os.path.exists(yolo_weights_pth_path):
            model = load_model(cfg_file_path, weights_path=yolo_weights_pth_path)
        else:
            model = load_model(cfg_file_path)
            
    
    trainer = AdversarialTrainer(
        model, 
        trainer_config['dataset_kwargs'],
        trainer_config['optimizer_kwargs'],
        trainer_config['trainer_kwargs'],
        trainer_config['lr_scheduler_kwargs']
    )
    
    trainer.train()    