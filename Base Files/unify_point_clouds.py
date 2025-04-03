import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def estimate_normals(pcd, radius=0.1, max_nn=30):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return pcd

def pairwise_registration(source, target):
    max_correspondence_distance = 0.02 * 1.5

    # Estimate normals if they don't exist
    if not source.has_normals():
        source = estimate_normals(source)
    if not target.has_normals():
        target = estimate_normals(target)

    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance, icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

parent_folder = "med2"
start_frame = 269
end_frame = 275
# Load the three point clouds
# pc1 = load_point_cloud(f"{parent_folder}/imu_point_cloud_{start_frame}_to_{end_frame}_248622301868_2025-01-24.ply")
# pc2 = load_point_cloud(f"{parent_folder}/imu_point_cloud_{start_frame}_to_{end_frame}_248622302627_2025-01-24.ply")
# pc3 = load_point_cloud(f"{parent_folder}/imu_point_cloud_{start_frame}_to_{end_frame}_336522303074_2025-01-24.ply")
source = load_point_cloud(f"{parent_folder}/unified_point_cloud_pc1_pc2.ply")
target = load_point_cloud(f"{parent_folder}/unified_point_cloud_pc1_pc3.ply")
# Estimate normals for all point clouds
pcds = [estimate_normals(pc) for pc in [source, target]]

# Full registration
max_correspondence_distance = 0.02
pose_graph = full_registration(pcds, max_correspondence_distance)

# Optimize the pose graph
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance,
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
    pcd_temp = pcd.transform(pose_graph.nodes[i].pose)
    pcd_combined += pcd_temp

# Downsample for uniform density
pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.0000001)

# Save and visualize the combined point cloud
o3d.io.write_point_cloud(f'{parent_folder}/unified_point_cloud.ply', pcd_combined)
o3d.visualization.draw_geometries([pcd_combined])
