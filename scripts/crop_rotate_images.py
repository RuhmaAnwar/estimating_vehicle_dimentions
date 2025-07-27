import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from calculate_scaling_factor import get_matched_annotations

def get_vehicle_corners(x, y, heading, length, width):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    half_length = length / 2
    half_width = width / 2
    corners = [(-half_length, half_width), (half_length, half_width),
               (half_length, -half_width), (-half_length, -half_width)]
    rotated_corners = np.zeros((4, 2))
    for i, (corner_x, corner_y) in enumerate(corners):
        rotated_corners[i, 0] = x + corner_x * cos_h - corner_y * sin_h
        rotated_corners[i, 1] = y + corner_x * sin_h + corner_y * cos_h
    return rotated_corners

def main():
    parent_dir = "/home/ruhma/estimating_vehicle_dimentions"
    output_dir = os.path.join(parent_dir, "rotated_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scaling factors
    scaling_factors_path = os.path.join(parent_dir, "results/scaling_factors.csv")
    if not os.path.exists(scaling_factors_path):
        raise FileNotFoundError(f"Scaling factors CSV not found at {scaling_factors_path}. Run calculate_scaling_factor.py first.")
    scaling_df = pd.read_csv(scaling_factors_path)
    
    for did in range(10):
        drone = f'20181029_D{did+1}_0900_0930'
        drone_h5 = f'20181029_d{did+1}_0900_0930'
        image_dir = os.path.join(parent_dir, f'RawDatasets/pNEUMA_Vision/{drone}/Frames')
        csv_dir = os.path.join(parent_dir, f'RawDatasets/pNEUMA_Vision/{drone}/Annotations')
        hdf_path = os.path.join(parent_dir, f'InputData/pNEUMA/d{did+1}/data_{drone_h5}.h5')
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        for image_file in tqdm(image_files, desc=f"Processing {drone}"):
            image_num = os.path.splitext(image_file)[0]
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image {image_path}")
                continue
            
            csv_path = os.path.join(csv_dir, f"{image_num}.csv")
            if not os.path.exists(csv_path):
                print(f"CSV file {csv_path} not found")
                continue
            
            # Get scaling factor for this image
            scaling_row = scaling_df[(scaling_df['drone'] == drone) & (scaling_df['image'] == image_file)]
            if scaling_row.empty:
                print(f"No scaling factor found for {drone}/{image_file}. Skipping.")
                continue
            scaling_factor = scaling_row['scaling_factor'].iloc[0]
            
            matched_df = get_matched_annotations(image_path, csv_path, hdf_path)
            if matched_df.empty:
                continue
            
            for _, row in tqdm(matched_df.iterrows(), total=len(matched_df), desc=f"Processing vehicles in {image_file}", leave=False):
                track_id = row['track_id']
                vehicle_type = row['agent_type']
                x = row['x_img']
                y = row['y_img']
                angle = row['angle_img']
                length = (row['length'] + 5) / scaling_factor  # Convert meters to pixels, add 5m
                width = (row['width'] + 2) / scaling_factor    # Convert meters to pixels, add 2m
                
                # Create vehicle type and track ID subfolders
                drone_id = f"d{did+1}"
		track_folder = os.path.join(output_dir, vehicle_type, f"{int(track_id)}_{drone_id}")

                os.makedirs(track_folder, exist_ok=True)
                
                corners = get_vehicle_corners(x, y, angle, length, width)
                corners = corners.astype(np.intp)
                
                x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
                y_min, y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
                
                height, width_img = image.shape[:2]
                x_min, x_max = max(x_min, 0), min(x_max, width_img)
                y_min, y_max = max(y_min, 0), min(y_max, height)
                
                if x_max <= x_min or y_max <= y_min:
                    print(f"Invalid bounding box for Track ID {track_id} ({vehicle_type}) in {image_file}")
                    continue
                
                cropped_image = image[y_min:y_max, x_min:x_max]
                crop_center_x = (x_max - x_min) / 2
                crop_center_y = (y_max - y_min) / 2
                angle_deg = math.degrees(angle)
                rotation_angle = 270 - angle_deg
                
                crop_height, crop_width = cropped_image.shape[:2]
                cos_a = abs(np.cos(math.radians(rotation_angle)))
                sin_a = abs(np.sin(math.radians(rotation_angle)))
                new_width = int(crop_width * cos_a + crop_height * sin_a)
                new_height = int(crop_width * sin_a + crop_height * cos_a)
                
                M = cv2.getRotationMatrix2D((crop_center_x, crop_center_y), rotation_angle, 1.0)
                M[0, 2] += (new_width - crop_width) / 2
                M[1, 2] += (new_height - crop_height) / 2
                
                rotated_image = cv2.warpAffine(cropped_image, M, (new_width, new_height))
                
                output_path = os.path.join(track_folder, f"{image_num}_{int(track_id)}.jpg")
                cv2.imwrite(output_path, rotated_image)

if __name__ == "__main__":
    main()

