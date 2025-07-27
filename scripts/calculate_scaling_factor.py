import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_matched_annotations(image_path, csv_path, hdf_path):
    annotations = pd.read_csv(csv_path)
    data = pd.read_hdf(hdf_path)
    matched_data = []
    for _, row in annotations.iterrows():
        track_id = row['ID']
        time = row['Time [s]']
        match = data[(data['track_id'] == track_id) & (data['time'] == time)]
        if not match.empty:
            matched_data.append({
                'track_id': track_id,
                'time': time,
                'agent_type': row['Type'],
                'x_img': row['x_img [px]'],
                'y_img': row['y_img [px]'],
                'angle_img': row['Angle_img [rad]'],
                'x': match['x'].iloc[0],
                'y': match['y'].iloc[0],
                'length': match['length'].iloc[0],
                'width': match['width'].iloc[0],
                'psi_kf': match['psi_kf'].iloc[0],
            })
    return pd.DataFrame(matched_data)

def calculate_scaling_factor(df):
    pixel_dists, real_dists = [], []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            dx_img = df['x_img'].iloc[i] - df['x_img'].iloc[j]
            dy_img = df['y_img'].iloc[i] - df['y_img'].iloc[j]
            pixel_dist = np.sqrt(dx_img**2 + dy_img**2)
            dx = df['x'].iloc[i] - df['x'].iloc[j]
            dy = df['y'].iloc[i] - df['y'].iloc[j]
            real_dist = np.sqrt(dx**2 + dy**2)
            if pixel_dist > 0:
                pixel_dists.append(pixel_dist)
                real_dists.append(real_dist)
    if pixel_dists:
        scaling_factors = np.array(real_dists) / np.array(pixel_dists)
        return np.mean(scaling_factors)
    return None

def main():
    parent_dir = "/home/ruhma/estimating_vehicle_dimentions"
    scaling_factors = []

    for did in range(1):  # Change to range(10) for all 10 days
        drone = f'20181029_D{did+1}_0900_0930'
        drone_h5 = f'20181029_d{did+1}_0900_0930'
        image_dir = os.path.join(parent_dir, f'RawDatasets/pNEUMA_Vision/{drone}/Frames')
        csv_dir = os.path.join(parent_dir, f'RawDatasets/pNEUMA_Vision/{drone}/Annotations')
        hdf_path = os.path.join(parent_dir, f'InputData/pNEUMA/d{did+1}/data_{drone_h5}.h5')

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        for image_file in tqdm(image_files, desc=f"Processing {drone}"):
            image_num = os.path.splitext(image_file)[0]
            image_path = os.path.join(image_dir, image_file)
            csv_path = os.path.join(csv_dir, f"{image_num}.csv")

            if not os.path.exists(csv_path):
                print(f"CSV file {csv_path} not found")
                continue

            matched_df = get_matched_annotations(image_path, csv_path, hdf_path)
            if matched_df.empty:
                continue

            scaling_factor = calculate_scaling_factor(matched_df)
            if scaling_factor is not None:
                scaling_factors.append({
                    'drone': drone,
                    'image': image_file,
                    'scaling_factor': scaling_factor
                })

    # Save scaling factors to CSV
    scaling_df = pd.DataFrame(scaling_factors)
    output_csv = os.path.join(parent_dir, 'results/scaling_factors.csv')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    scaling_df.to_csv(output_csv, index=False)
    print(f"Saved scaling factors to {output_csv}")
    print(f"Average scaling factor: {scaling_df['scaling_factor'].mean():.6f} meters/pixel")

if __name__ == "__main__":
    main()

