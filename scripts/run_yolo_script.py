from ultralytics import YOLO
import os
import numpy as np
from tqdm import tqdm

# Load the YOLO11 model
model = YOLO("yolo11x.pt")

# Define input and output directories
parent_dir = "/home/ruhma/estimating_vehicle_dimentions"
input_folder = os.path.join(parent_dir, "rotated_images")
output_folder = os.path.join(parent_dir, "yolo_output")

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of vehicle types
vehicle_types = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

# Process each drone
for did in range(10):
    drone = f'20181029_D{did+1}_0900_0930'
    drone_id = f"d{did+1}"
    
    # Initialize progress bar for images across all vehicle types for this drone
    image_files = []
    for vehicle_type in vehicle_types:
        vehicle_dir = os.path.join(input_folder, vehicle_type)
        track_dirs = [d for d in os.listdir(vehicle_dir) if os.path.isdir(os.path.join(vehicle_dir, d)) and d.endswith(f"_{drone_id}")]
        for track_dir in track_dirs:
            track_path = os.path.join(vehicle_dir, track_dir)
            files = [f for f in os.listdir(track_path) if f.lower().endswith(image_extensions)]
            for f in files:
                image_files.append((vehicle_type, track_dir, f))
    
    for vehicle_type, track_dir, image_file in tqdm(image_files, desc=f"Processing images for {drone}"):
        img_path = os.path.join(input_folder, vehicle_type, track_dir, image_file)
        
        # Create output subfolder
        output_subfolder = os.path.join(output_folder, vehicle_type, track_dir)
        os.makedirs(output_subfolder, exist_ok=True)
        
        base_filename = os.path.basename(img_path)
        output_txt_path = os.path.join(output_subfolder, f"pred_{base_filename.rsplit('.', 1)[0]}.txt")
        
        # Skip if already processed
        if os.path.exists(output_txt_path):
            continue
        
        # Run prediction on one image
        results = model.predict(img_path, conf=0.1, stream=False, verbose=False)

        
        # Process result (single image, so results[0])
        result = results[0]
        
        # Save annotated image
        output_image_path = os.path.join(output_subfolder, f"pred_{base_filename}")
        result.save(filename=output_image_path)
        
        # Save detection results as a text file (empty if no detections)
        with open(output_txt_path, 'w') as f:
            if result.boxes:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    xywh = box.xywhn[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    f.write(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {conf:.6f}\n")
