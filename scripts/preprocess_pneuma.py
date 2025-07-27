import os
import csv
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed

# Credit: Adapted from https://github.com/Yiru-Jiao/DriverSpaceInference (Jiao et al., 2023)
# Citation: Jiao, Y., Calvert, S. C., van Cranenburgh, S., & van Lint, H. (2023). 
# Inferring vehicle spacing in urban traffic from trajectory data. 
# Transportation Research Part C: Emerging Technologies, 155, 104289. 
# DOI: 10.1016/j.trc.2023.104289

parent_dir = "/home/ruhma/estimating_vehicle_dimentions"

def pNEUMA(open_path, data_title):
    data_path = os.path.join(open_path, f"{data_title}.csv")
    data_overview_cols = ['track_id', 'type', 'traveled_d', 'avg_speed']
    data_cols = ['lat', 'lon', 'speed', 'lon_acc', 'lat_acc', 'time', 'track_id']
    
    data_overview_rows = []
    data_rows = []
    
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        next(csv_reader)
        for row in csv_reader:
            row = [item.strip() for item in row]
            track_id = int(row[0])
            data_overview_rows.append(row[:len(data_overview_cols)])
            
            data_traj = [float(item) for item in row[len(data_overview_cols):] if item]
            for i in range(0, len(data_traj), len(data_cols)-1):
                data_row = data_traj[i:i+len(data_cols)-1] + [track_id]
                data_rows.append(data_row)
                
    data_overview = pd.DataFrame(data_overview_rows, columns=data_overview_cols)
    data_overview['track_id'] = data_overview['track_id'].astype(int)
    data_overview = data_overview.set_index('track_id')

    # Add vehicle size
    length = np.zeros(data_overview.shape[0])
    width = np.zeros(data_overview.shape[0])
    for i in range(data_overview.shape[0]):
        vehicle_type = data_overview['type'].iloc[i]
        if vehicle_type == 'Motorcycle':
            length[i], width[i] = 2.5, 1
        elif vehicle_type in ['Car', 'Taxi']:
            length[i], width[i] = 5, 2
        elif vehicle_type == 'Medium Vehicle':
            length[i], width[i] = 5.83, 2.67
        elif vehicle_type == 'Heavy Vehicle':
            length[i], width[i] = 12.5, 3.33
        elif vehicle_type == 'Bus':
            length[i], width[i] = 12.5, 4
    data_overview['length'] = length
    data_overview['width'] = width
    
    data = pd.DataFrame(data_rows, columns=data_cols)
    time_interval = data['time'].iloc[1] - data['time'].iloc[0]
    data['frame_id'] = round(data['time'] / time_interval).astype(int)
    data.set_index(['track_id', 'frame_id'], drop=True, inplace=True)
    
    data_overview[['traveled_d', 'avg_speed']] = data_overview[['traveled_d', 'avg_speed']].astype(float)
    data_overview['avg_speed'] = data_overview['avg_speed'] / 3.6
    data['speed'] = data['speed'] / 3.6
    
    utm_crs_list = query_utm_crs_info(
        datum_name='WGS84',
        area_of_interest=AreaOfInterest(
            west_lon_degree=np.floor(data['lon'].min()), 
            south_lat_degree=np.floor(data['lat'].min()), 
            east_lon_degree=np.ceil(data['lon'].max()), 
            north_lat_degree=np.ceil(data['lat'].max())
        )
    )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    crs = CRS.from_epsg(4326)
    geo_coordinates = Transformer.from_crs(crs.geodetic_crs, utm_crs)
    x, y = geo_coordinates.transform(data['lat'].values, data['lon'].values)
    data['x'] = x - 739000
    data['y'] = y - 4206500

    return data_overview, data

def KFV(track_id, data):
    veh = data.loc[track_id].copy()
    numstates = 4
    P = np.eye(numstates) * 50
    dt = np.diff(veh.time, prepend=veh.time.iloc[0])
    R = np.diag([3.0, 3.0, 2.0])
    I = np.eye(numstates)
    mx, my, mv = veh.x.values, veh.y.values, veh.speed.values

    measurements = np.vstack((mx, my, mv))
    m = measurements.shape[1]
    head = veh[['x', 'y']].diff(10).dropna()
    head = head[(head.x != 0) | (head.y != 0)].values
    
    if len(head) == 0:
        estimates = np.zeros((m, 4)) * np.nan
    else:
        psi0 = float(np.arctan2(head[0][1], head[0][0]))
        x = np.array([[float(mx[0]), float(my[0]), float(mv[0]), psi0]]).T
        estimates = np.zeros((m, 4))

        for filterstep in range(m):
            x2_scalar = float(x[2, 0])
            x3_scalar = float(x[3, 0])
            x[0, 0] += dt[filterstep] * x2_scalar * np.cos(x3_scalar)
            x[1, 0] += dt[filterstep] * x2_scalar * np.sin(x3_scalar)
            x[3, 0] = (x3_scalar + np.pi) % (2.0 * np.pi) - np.pi

            a13 = dt[filterstep] * np.cos(x3_scalar)
            a14 = -dt[filterstep] * x2_scalar * np.sin(x3_scalar)
            a23 = dt[filterstep] * np.sin(x3_scalar)
            a24 = dt[filterstep] * x2_scalar * np.cos(x3_scalar)
            JA = np.array([[1.0, 0.0, a13, a14],
                           [0.0, 1.0, a23, a24],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

            sGPS = 4 * dt[filterstep] ** 2
            sCourse = 1.0 * dt[filterstep]
            sVelocity = 18.0 * dt[filterstep]
            Q = np.diag([sGPS**2, sGPS**2, sVelocity**2, sCourse**2])

            P = JA @ P @ JA.T + Q
            hx = np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]]])
            JH = np.eye(3, 4)
            S = JH @ P @ JH.T + R
            K = (P @ JH.T) @ np.linalg.inv(S)
            Z = measurements[:, filterstep].reshape(3, 1)
            y = Z - hx
            x += K @ y
            P = (I - K @ JH) @ P
            estimates[filterstep, :] = x.T

    veh[['x_kf', 'y_kf', 'speed_kf', 'psi_kf']] = estimates
    veh['track_id'] = track_id
    return veh.reset_index()

def zeroheading(df):
    if np.all((abs(df['vx_kf'].iloc[5:]) <= 0.5) & (abs(df['vy_kf'].iloc[5:]) <= 0.5)):
        return True
    return False

def cleaning(data):
    remove_list = data.groupby('track_id').apply(zeroheading)
    remove_list = remove_list.index[remove_list]
    data = data.set_index(['track_id', 'frame_id'])
    data = data.drop(remove_list, level=0)
    print(f'Removed {len(remove_list)} stationary vehicles')
    return data, remove_list

def main():
    parent_dir = "/home/ruhma/estimating_vehicle_dimentions"
    for dx in [f'd{did+1}' for did in range(10)]:
        data_files = sorted(glob.glob(os.path.join(parent_dir, f'RawDatasets/pNEUMA/{dx}/*.csv')))
        open_path = os.path.join(parent_dir, f'RawDatasets/pNEUMA/{dx}')
        save_path = os.path.join(parent_dir, f'InputData/pNEUMA/{dx}')
        os.makedirs(save_path, exist_ok=True)

        data_titles = [os.path.splitext(os.path.basename(file))[0] for file in data_files]
        for data_title in data_titles:
            save_file = os.path.join(save_path, f'data_{data_title}.h5')
            if os.path.exists(save_file):
                print(f"Skipping {data_title}: already processed.")
                continue

            print(f"---- Preprocessing {data_title} ----")
            data_overview, data = pNEUMA(open_path, data_title)
            track_ids = data.reset_index().groupby('track_id').frame_id.count()
            track_ids = track_ids[track_ids >= 25].index.values
            data = data.loc[track_ids]
            data = pd.concat(Parallel(n_jobs=25)(delayed(KFV)(id, data) for id in tqdm(track_ids)))
            data['vx_kf'] = data.speed_kf * np.cos(data.psi_kf)
            data['vy_kf'] = data.speed_kf * np.sin(data.psi_kf)

            data, nevermove = cleaning(data)
            if len(nevermove) > 0:
                pd.DataFrame(nevermove).to_csv(os.path.join(save_path, f'nevermove_{data_title}.csv'), index=False)

            data[['length', 'width']] = data_overview.reindex(index=data.index.get_level_values(0))[['length', 'width']].values
            data['agent_type'] = data_overview.reindex(index=data.index.get_level_values(0))['type'].values
            data.reset_index().to_hdf(save_file, key='data')

if __name__ == "__main__":
    main()
