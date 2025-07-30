# Estimating Vehicle Dimensions from pNEUMA Vision Dataset using YOLO11x

This repository processes the pNEUMA Vision dataset to detect vehicle dimentions (Motorcycle, Car, Taxi, Medium Vehicle, Heavy Vehicle, Bus) from drone images from October 29, 2018, using YOLO11. The pipeline preprocesses pNEUMA trajectory data, calculates per-image scaling factors, crops and rotates vehicle images, runs YOLO11x detection, removes empty detections, and generates a CSV summary of results.

## Citations

Please cite:

- **pNEUMA Vision**: Barmpounakis, E., et al. (2022). A new dataset for multi-modal traffic analysis with large-scale vehicle trajectories and drone images. *Transportation Research Part C: Emerging Technologies*, 135, 103966. DOI: 10.1016/j.trc.2022.103966
- **Preprocessing Code**: Adapted from Jiao, Y., Calvert, S. C., van Cranenburgh, S., & van Lint, H. (2023). Inferring vehicle spacing in urban traffic from trajectory data. *Transportation Research Part C: Emerging Technologies*, 155, 104289. DOI: 10.1016/j.trc.2023.104289
- **pNEUMA Dataset**: Barmpounakis, E., & Geroliminis, N. (2020). pNEUMA dataset [Data set]. In On the new era of urban traffic monitoring with massive drone data: The pNEUMA large-scale field experiment, *Transportation Research Part C: Emerging Technologies*, 111, 50–71. DOI: 10.5281/zenodo.10491409

## Prerequisites

- **System**: Linux (tested on Ubuntu)
- **Hardware**: CPU (GPU recommended for faster YOLO11 inference)
- **Software**: Miniconda, Python 3.12
- **Data**:
  - pNEUMA Vision Dataset: https://zenodo.org/records/7426506#.Y-D-Ky8w1hE
  - pNEUMA Dataset: https://open-traffic.epfl.ch/index.php/downloads/#1599047632394-7ca81bff-5221

## Directory Structure

```
/home/ruhma/estimating_vehicle_dimentions/
├── RawDatasets/
│   ├── pNEUMA/
│   │   ├── d1/
│   │   │   ├── 20181029_d1_0900_0930.csv
│   │   │   └── ...
│   │   ├── d2/
│   │   └── ...
│   ├── pNEUMA_Vision/
│   │   ├── 20181029_D1_0900_0930/
│   │   │   ├── Annotations/
│   │   │   │   ├── 00001.csv
│   │   │   │   └── ...
│   │   │   ├── Frames/
│   │   │   │   ├── 00001.jpg
│   │   │   │   └── ...
│   │   ├── 20181029_D2_0900_0930/
│   │   └── ...
├── InputData/
│   ├── pNEUMA/
│   │   ├── d1/
│   │   │   ├── data_20181029_d1_0900_0930.h5
│   │   │   ├── nevermove_20181029_d1_0900_0930.csv
│   │   │   └── ...
│   │   ├── d2/
│   │   └── ...
├── rotated_images/
│   ├── Car/
│   │   ├── 21_d1/
│   │   │   ├── 00001_21.jpg
│   │   │   └── ...
│   │   ├── 22_d2/
│   │   └── ...
│   ├── Taxi/
│   ├── Motorcycle/
│   ├── Medium Vehicle/
│   ├── Heavy Vehicle/
│   ├── Bus/
│   └── ...
├── yolo_output/
│   ├── Car/
│   │   ├── 21_d1/
│   │   │   ├── pred_00001_21.jpg
│   │   │   ├── pred_00001_21.txt
│   │   │   └── ...
│   │   ├── 22_d2/
│   │   └── ...
│   ├── Taxi/
│   ├── Motorcycle/
│   ├── Medium Vehicle/
│   ├── Heavy Vehicle/
│   ├── Bus/
│   └── ...
├── results/
│   ├── detections.csv
│   ├── scaling_factors.csv
│   ├── sample_images/
│   └── ...
├── scripts/
│   ├── setup_environment.sh
│   ├── preprocess_pneuma.py
│   ├── calculate_scaling_factor.py
│   ├── crop_rotate_images.py
│   ├── run_yolo_script.py
│   ├── delete_empty_detections.py
│   ├── generate_csv.py
├── README.md
└── LICENSE
```

## Setup Instructions

1. **Clone Repository**:

   ```bash
   git clone https://github.com/RuhmaAnwar/estimating_vehicle_dimentions.git
   cd estimating_vehicle_dimentions
   ```

2. **Download Data**:

   - Download pNEUMA Vision data from Zenodo and extract to `RawDatasets/pNEUMA_Vision/`.
   - Download pNEUMA trajectory data from Open Traffic and extract to `RawDatasets/pNEUMA/`.

3. **Set Up Environment**:

   ```bash
   chmod +x scripts/setup_environment.sh
   ./scripts/setup_environment.sh
   source ~/.bashrc
   conda activate yolo_env
   ```

4. **Increase File Descriptor Limit** (to avoid "Too many open files" error):

   ```bash
   ulimit -n 4096
   ```

## Pipeline Steps

Run the following scripts in order from the `estimating_vehicle_dimentions` directory:

1. **Preprocess pNEUMA Trajectory Data**:

   ```bash
   python scripts/preprocess_pneuma.py
   ```

   - **Output**: `InputData/pNEUMA/d*/data_20181029_d*_0900_0930.h5`, `nevermove_*.csv`
   - Processes trajectory data, applies Kalman filtering, and adds vehicle dimensions.

2. **Calculate Scaling Factors**:

   ```bash
   python scripts/calculate_scaling_factor.py
   ```

   - **Output**: `results/scaling_factors_d<drone_id>.csv`
   - Computes per-image scaling factors (meters/pixel) for drones D1–D10.

3. **Crop and Rotate Images**:

   ```bash
   python scripts/crop_rotate_images.py
   ```

   - **Output**: `rotated_images/<vehicle_type>/<track_id>_d<drone_id>/<image_num>_<track_id>.jpg`
   - Crops and rotates vehicle images using per-image scaling factors and vehicle-specific dimensions (with +5m length, +2m width).

4. **Run YOLO11 Predictions**:

   ```bash
   python scripts/run_yolo_script.py
   ```

   - **Output**: `yolo_output/<vehicle_type>/<track_id>_d<drone_id>/pred_<image_num>_<track_id>.jpg`, `.txt`
   - Applies YOLO11x model (confidence threshold 0.1) to detect objects in cropped images.

5. **Delete Empty Detections**:

   ```bash
   python scripts/delete_empty_detections.py
   ```

   - **Output**: Removes empty `.txt` files and corresponding images from `yolo_output/`.

6. **Generate CSV Summary**:

   ```bash
   python scripts/generate_csv.py
   ```

   - **Output**: `results/detections.csv`
   - Compiles detection results (image path, vehicle type, track ID, class, bounding box, confidence).

## Outputs

- **Preprocessed Data**: `InputData/pNEUMA/d*/data_*.h5` (HDF5 files with trajectory data and vehicle dimensions)
- **Scaling Factors**: `results/scaling_factors_d<drone_id>.csv` (drone, image, scaling factor)
- **Cropped Images**: `rotated_images/<vehicle_type>/<track_id>_d<drone_id>/<image_num>_<track_id>.jpg`
- **YOLO Outputs**: `yolo_output/<vehicle_type>/<track_id>_d<drone_id>/pred_*.jpg`, `pred_*.txt`
- **CSV Summary**: `results/detections.csv` (detection details)
- **Sample Images**: `results/sample_images/` (example YOLO outputs)

## Notes

- **YOLO11 Model**: The pipeline uses `yolo11x.pt`, which may detect non-vehicle classes (e.g., "bottle", "vase"); however since we were only interested in vehicle dimentions, classes were ignored as long as the vehicle was outlined. 

  ```
- **Performance**: Use a GPU for faster YOLO inference .
- **Data Size**: The pipeline generates many images. Ensure sufficient disk space (\~50GB for all drones).

## Troubleshooting

- **File Descriptor Error**:

  ```bash
  ulimit -n 4096
  ```
- **Invalid Bounding Boxes**:
  - Check image dimensions and annotations:

    ```bash
    xdg-open RawDatasets/pNEUMA_Vision/20181029_D1_0900_0930/Frames/00001.jpg
    cat RawDatasets/pNEUMA_Vision/20181029_D1_0900_0930/Annotations/00001.csv
    ```
## Acknowledgment

**Data source**: pNEUMA – open-traffic.epfl.ch

## Contact

For issues, open a GitHub issue or contact ruhmaanwar16@gmail.com

## License

This project is licensed under the MIT License. See the LICENSE file for details.
