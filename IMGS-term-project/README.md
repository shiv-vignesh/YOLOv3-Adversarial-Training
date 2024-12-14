# Adversarial Training of YOLOv3 on KiTTi Dataset 

This project implements a trainer framework for training YOLOv3 on FGSM and PGD robustness attack. 

## Usage 

1. Prepare the KITTI dataset:
    - Download the KITTI object detection dataset
    - Organize the data in the following structure:

            data/
                ├── calibration/
                │   └── training/
                │       └── calib/
                ├── left_images/
                │   └── training/
                │       └── image_2/
                ├── labels/
                │   └── training/
                │       └── label_2/
                └── velodyne/
                    └── training/
                        └── velodyne/

2. Train the model:

```bash
python train_adv.py 
```

3. Evaluate the model:

```bash
python test_yolo_adv.py
```

## Configuration File: `config/yolo_adv_trainerjson`

### 1. **Dataset Configuration (`dataset_kwargs`)**

#### Training Dataset
Specify paths and parameters for training & Validation data:
```json
"trainer_dataset_kwargs": {
    "lidar_dir": "data/velodyne/training/velodyne",
    "calibration_dir": "data/calibration/training/calib",
    "left_image_dir": "data/left_images/training/image_2",
    "labels_dir": "data/labels/training/label_2",
    "shuffle": true,
    "apply_augmentation": false
}

"validation_dataset_kwargs": {
    "lidar_dir": "data/velodyne/validation/velodyne",
    "calibration_dir": "data/calibration/validation/calib",
    "left_image_dir": "data/left_images/validation/image_2",
    "labels_dir": "data/labels/validation/label_2",
    "shuffle": false,
    "apply_augmentation": false
}
``` 

## Trainer Configuration (`trainer_kwargs`)

The `trainer_kwargs` section in the JSON configuration file controls the behavior of the training process. Below is an explanation of each key parameter.

### Key Parameters

1. **`output_dir`**
   - Specifies the directory where training outputs (e.g., model checkpoints, logs) will be saved.
   - Example:  
     ```json
     "output_dir": "Robust-Spatial-Fusion-Pipeline-3"
     ```
