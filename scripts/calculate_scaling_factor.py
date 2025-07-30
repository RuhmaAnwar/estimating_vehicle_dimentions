import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_matched_annotations(image_path: Path, csv_path: Path, hdf_path: Path) -> pd.DataFrame:
    """Match annotations from CSV and HDF data based on track_id and time."""
    annotations = pd.read_csv(csv_path)
    data = pd.read_hdf(hdf_path)
    
    # Use merge for faster matching instead of iterative filtering
    matched = annotations.merge(
        data,
        left_on=['ID', 'Time [s]'],
        right_on=['track_id', 'time'],
        how='inner'
    )
    
    return pd.DataFrame({
        'track_id': matched['ID'],
        'time': matched['Time [s]'],
        'agent_type': matched['Type'],
        'x_img': matched['x_img [px]'],
        'y_img': matched['y_img [px]'],
        'angle_img': matched['Angle_img [rad]'],
        'x': matched['x'],
        'y': matched['y'],
        'length': matched['length'],
        'width': matched['width'],
        'psi_kf': matched['psi_kf']
    })

def calculate_scaling_factor(df: pd.DataFrame) -> float:
    """Calculate the mean scaling factor from pixel to real-world distances."""
    if len(df) < 2:
        return None
        
    # Vectorized distance calculations
    x_img, y_img = df['x_img'].values, df['y_img'].values
    x, y = df['x'].values, df['y'].values
    
    # Create all pairwise differences
    n = len(df)
    idx = np.triu_indices(n, k=1)  # Upper triangle indices to avoid duplicates
    dx_img = x_img[idx[0]] - x_img[idx[1]]
    dy_img = y_img[idx[0]] - y_img[idx[1]]
    dx = x[idx[0]] - x[idx[1]]
    dy = y[idx[0]] - y[idx[1]]
    
    # Calculate distances
    pixel_dists = np.sqrt(dx_img**2 + dy_img**2)
    real_dists = np.sqrt(dx**2 + dy**2)
    
    # Filter out zero pixel distances and compute scaling factors
    valid = pixel_dists > 0
    if valid.any():
        return np.mean(real_dists[valid] / pixel_dists[valid])
    return None

def main():
    """Process drone data to calculate and save scaling factors."""
    parent_dir = Path("/home/ruhma/estimating_vehicle_dimentions")
    
    for did in range(10): 
        drone = f'20181029_D{did+1}_0900_0930'
        drone_id = f'd{did+1}'
        output_csv = parent_dir / f'results/scaling_factors_{drone_id}.csv'
        
        # Skip if output already exists
        if output_csv.exists():
            print(f"Skipping {drone}: {output_csv.name} already exists")
            continue
        
        # Define paths
        drone_h5 = f'20181029_d{did+1}_0900_0930'
        image_dir = parent_dir / f'RawDatasets/pNEUMA_Vision/{drone}/Frames'
        csv_dir = parent_dir / f'RawDatasets/pNEUMA_Vision/{drone}/Annotations'
        hdf_path = parent_dir / f'InputData/pNEUMA/d{did+1}/data_{drone_h5}.h5'
        
        # Validate paths early
        if not (image_dir.exists() and csv_dir.exists() and hdf_path.exists()):
            print(f"Data for {drone} not found (missing image_dir, csv_dir, or hdf_path)")
            continue
        
        # Get image files
        image_files = [f for f in image_dir.glob('*.jpg')]
        if not image_files:
            print(f"No image files found in {image_dir}")
            continue
        
        scaling_factors = []
        for image_path in tqdm(image_files, desc=f"Processing {drone_id} images"):
            image_num = image_path.stem
            csv_path = csv_dir / f"{image_num}.csv"
            
            if not csv_path.exists():
                print(f"CSV file {csv_path} not found")
                continue
            
            matched_df = get_matched_annotations(image_path, csv_path, hdf_path)
            if matched_df.empty:
                continue
            
            scaling_factor = calculate_scaling_factor(matched_df)
            if scaling_factor is not None:
                scaling_factors.append({
                    'drone': drone,
                    'image': image_path.name,
                    'scaling_factor': scaling_factor
                })
        
        # Save results
        if scaling_factors:
            scaling_df = pd.DataFrame(scaling_factors)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            scaling_df.to_csv(output_csv, index=False)
            print(f"Saved scaling factors to {output_csv}")
        else:
            print(f"No scaling factors calculated for {drone}")

if __name__ == "__main__":
    main()
