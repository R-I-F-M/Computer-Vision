from ultralytics import YOLO
import cv2
import os

# Path to the fine-tuned model (YOLOv8n, YOLOv8s, etc.)
#model_path = '/content/runs/detect/yolo_finetuned/weights/best.pt'  # Use your trained model's path
model_path = "/content/best (2).pt"
test_img_dir = '/content/drive/MyDrive/updated_images'#'/content/plant/images'  # Path to the folder containing test images
output_dir = '/content/output'       # Folder to save output images with detections

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the fine-tuned YOLOv8 model
model = YOLO(model_path)

# Loop through each image in the test folder
for img_file in os.listdir(test_img_dir):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
        img_path = os.path.join(test_img_dir, img_file)

        # Perform inference on the test image
        results = model.predict(source=img_path, conf=0.7,imgsz=640)  # Confidence threshold can be adjusted

        # Loop through each result (there could be multiple images, but in this case just one)
        for i, result in enumerate(results):
            # Create the full path for saving the output
            output_img_path = os.path.join(output_dir, f"result_{i}_{img_file}")

            # Save the image with bounding boxes and labels
            result.save(filename=output_img_path)

        print(f"Tested {img_file} and saved the result to {output_dir}")

print("Testing completed, check the output folder for results.")