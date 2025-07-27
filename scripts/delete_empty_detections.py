import os
from tqdm import tqdm

# Define output directory
parent_dir = "/home/ruhma/estimating_vehicle_dimentions"
output_folder = os.path.join(parent_dir, "yolo_output")

# Get list of vehicle types
vehicle_types = [d for d in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, d))]

# Process each drone
for did in range(10):
    drone = f'20181029_D{did+1}_0900_0930'
    drone_id = f"d{did+1}"
    
    # Initialize progress bar for text files across all vehicle types for this drone
    txt_files = []
    for vehicle_type in vehicle_types:
        vehicle_dir = os.path.join(output_folder, vehicle_type)
        track_dirs = [d for d in os.listdir(vehicle_dir) if os.path.isdir(os.path.join(vehicle_dir, d)) and d.endswith(f"_{drone_id}")]
        for track_dir in track_dirs:
            track_path = os.path.join(vehicle_dir, track_dir)
            files = [f for f in os.listdir(track_path) if f.endswith(".txt")]
            for f in files:
                txt_files.append((vehicle_type, track_dir, f))
    
    for vehicle_type, track_dir, txt_file in tqdm(txt_files, desc=f"Checking empty detections for {drone}"):
        txt_path = os.path.join(output_folder, vehicle_type, track_dir, txt_file)
        # Check if text file is empty
        if os.path.getsize(txt_path) == 0:
            # Delete the text file
            os.remove(txt_path)
            
            # Delete the corresponding image file
            image_path = txt_path.replace(".txt", ".jpg")
            if os.path.exists(image_path):
                os.remove(image_path)
            else:
                # Try other extensions
                for ext in (".jpeg", ".png", ".bmp"):
                    alt_image_path = txt_path.replace(".txt", ext)
                    if os.path.exists(alt_image_path):
                        os.remove(alt_image_path)
                        break