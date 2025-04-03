import open3d as o3d
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R

def load_filtered_csv_data(csv_file_path, start_frame=428, end_frame=528, serial_number=336522303608):
    df = pd.read_csv(csv_file_path)
    print(df.iloc[0]['serial_number'])
    filtered_df = df[(df['frame_number'] >= start_frame) & 
                     (df['frame_number'] <= end_frame) &
                     (df['serial_number'] == serial_number)]
    return filtered_df.sort_values('frame_number')

def process_imu_data(imu_data, prev_position, prev_orientation, dt):
    acceleration = np.array([imu_data['accel_x'], imu_data['accel_y'], imu_data['accel_z']])
    gyroscope = np.array([imu_data['gyro_x'], imu_data['gyro_y'], imu_data['gyro_z']])
    
    # Update orientation
    gyro_rotation = R.from_rotvec(gyroscope * dt)
    new_orientation = prev_orientation * gyro_rotation
    
    # Update position
    acceleration_world = new_orientation.apply(acceleration)
    acceleration_world[2] -= 9.81  # Remove gravity
    new_position = prev_position + prev_orientation.apply(acceleration) * dt**2 / 2
    
    return new_position, new_orientation

def create_point_cloud(rgb_image_path, depth_image_path, intrinsic):
    if not os.path.exists(rgb_image_path) or not os.path.exists(depth_image_path):
        print(f"Error: Image files not found")
        return None

    color_raw = o3d.io.read_image(rgb_image_path)
    depth_raw = o3d.io.read_image(depth_image_path)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000, depth_trunc=3.0, convert_rgb_to_intensity=False)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd

def pairwise_registration(source, target):
    threshold = 0.02
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

# Load the filtered CSV data
parent_folder = "med2"
date_string = "2025-01-24"
csv_file_path = os.path.join(parent_folder, f"frame_metadata_{date_string}.csv")
# serial_number = 248622302627
# serial_number = 336522303608
serial_number = 248622301868
# serial_number = 336522303074
start_frame = 269
end_frame = 275

filtered_data = load_filtered_csv_data(csv_file_path, start_frame=start_frame, end_frame=end_frame, serial_number=serial_number)

position = np.zeros(3)
orientation = R.from_quat([0, 0, 0, 1])  # Identity quaternion
prev_timestamp = filtered_data.iloc[0]['rgb_timestamp']

# Create point clouds from images
pcds = []

for _, row in filtered_data.iterrows():
    frame_id = int(row['frame_number'])
    serial_number = int(row['serial_number'])
    print(f"reading {frame_id}")
    rgb_path = os.path.join(parent_folder, f'rgb_images_{date_string}/rgb_frame_{frame_id}_cam{serial_number}.png')
    depth_path = os.path.join(parent_folder, f'depth_images_{date_string}/depth_frame_{frame_id}_cam{serial_number}.png')
    
    timestamp = row['rgb_timestamp']
    dt = timestamp - prev_timestamp
    # position, orientation = process_imu_data(row, position, orientation, dt)
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=480, height=640, 
        fx=row['fx'], fy=row['fy'], 
        cx=row['ppx'], cy=row['ppy'])
    
    pcd = create_point_cloud(rgb_path, depth_path, intrinsic)
    if pcd is not None:
        # # Apply IMU-based transformation to the point cloud
        # imu_transform = np.eye(4)
        # imu_transform[:3, :3] = orientation.as_matrix()
        # imu_transform[:3, 3] = position
        # pcd.transform(imu_transform)
        pcds.append(pcd)

    prev_timestamp = timestamp

# Build pose graph
pose_graph = o3d.pipelines.registration.PoseGraph()
odometry = np.identity(4)
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

for i in range(len(pcds) - 1):
    trans = pairwise_registration(pcds[i], pcds[i + 1])
    odometry = np.dot(trans, odometry)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, i + 1, trans, uncertain=False))

# Optimize the pose graph
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=0.02,
    edge_prune_threshold=0.25,
    reference_node=0)
o3d.pipelines.registration.global_optimization(
    pose_graph,
    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    option)

# Transform and merge point clouds
pcd_combined = o3d.geometry.PointCloud()
for i, pcd in enumerate(pcds):
    pcd.transform(pose_graph.nodes[i].pose)
    pcd_combined += pcd

# Downsample for uniform density
# pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)

# Save and visualize the combined point cloud
o3d.io.write_point_cloud(f'{parent_folder}/imu_point_cloud_{start_frame}_to_{end_frame}_{serial_number}_{date_string}.ply', pcd_combined)
o3d.visualization.draw_geometries([pcd_combined])
