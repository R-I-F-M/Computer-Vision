import yaml

# data = {'train' :  '/content/plant',
#         'val' :  '/content/plant',
#         'nc': 1,
#         'names': ['Unhealthy Plant']
#         }

# # overwrite the data to the .yaml file
# with open('custom_data.yaml', 'w') as f:
#     yaml.dump(data, f)

# # read the content in .yaml file
# with open('custom_data.yaml', 'r') as f:
#     hamster_yaml = yaml.safe_load(f)
#     display(hamster_yaml)








from ultralytics import YOLO
import os

# Define paths for the images and annotations
# train_img_dir = '/content/plant/imggg'  # Path to your training images folder
# train_ann_dir = '/content/plant/ann'  # Path to your YOLO annotations (txt files)
yaml_path = 'custom_data.yaml'  # Path to your YAML file for YOLO training
# # Create a YAML file for YOLO training
# data_yaml = f"""
# train: {train_img_dir},  # Path to training images
# val: {train_img_dir},    # Using the same folder for validation in this case (can be changed)
# nc: 2,                   # Number of classes (modify according to your dataset)
# names: ['healthy', 'stressed']  # List of class names
# """

# # Save the data configuration as a YAML file
# yaml_path = '/content/custom_data.yaml'
# with open(yaml_path, 'w') as f:
#     f.write(data_yaml)

# Load the pre-trained YOLOv8 model for fine-tuning
model = YOLO('yolo11n.pt')  # You can choose other YOLOv8 models (e.g., yolov8s, yolov8m, etc.)

# Fine-tune the YOLOv8 model on your dataset
model.train(
    data=yaml_path,       # Path to the YAML file we just created
    epochs=500,           # Number of training epochs
    imgsz=640,            # Image size (can be modified)
    batch=16,             # Batch size (adjust based on your GPU capacity)
    name='yolo_finetuned', # Name of the experiment
    device=0              # Use GPU (0) or CPU (-1)
)

# Evaluate the model after training
metrics = model.val()

# Optionally export the model in ONNX format for deployment
model.export(format='onnx')