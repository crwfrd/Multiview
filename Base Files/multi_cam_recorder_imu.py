import pyrealsense2 as rs
from multiprocessing import Process, Queue, freeze_support
import cv2
import numpy as np
import csv
import os
import argparse
import shutil
from datetime import datetime
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-camera recorder with IMU data")
    parser.add_argument("parent_folder", type=str, help="Parent folder name for storing data")
    return parser.parse_args()


# Define a function for capturing frames from a single camera
def capture_frames(serial_number, queue):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    
    profile = pipeline.start(config)
    
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    
    # Store frame number
    frame_number = 0
    
    # Create an align object to align depth to color for each pipeline
    align = rs.align(rs.stream.color)

    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            
            # Align the colour and depth frames
            aligned_frames = align.process(frames)
            aligned_frames.keep()
            
            # Get aligned depth and colour frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            
            # If frames could not be retrieved
            if not color_frame or not depth_frame or not accel_frame or not gyro_frame:
                continue
            
            # filtered_depth = spatial_filter.process(depth_frame)
            # filtered_depth = temporal_filter.process(filtered_depth)
            
            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            
            # Get camera intrinsics
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            fx, fy, ppx, ppy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

            # Get IMU data
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            
            accel_x, accel_y, accel_z = accel_data.x, accel_data.y, accel_data.z
            gyro_x, gyro_y, gyro_z = gyro_data.x, gyro_data.y, gyro_data.z
            
            # Retrieve timestamps
            rgb_timestamp = color_frame.get_timestamp()
            depth_timestamp = depth_frame.get_timestamp()
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            frame_number += 1

            # Send frames and IMU data to the queue
            queue.put((serial_number, color_image, depth_image, depth_colormap, rgb_timestamp, depth_timestamp, frame_number, fx, fy, ppx, ppy, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z))
    finally:
        pipeline.stop()
    

# Process the frames from the queue
def process_frames(queue, csv_filename, date_string, num_cameras, parent_folder):
    while True:
        # Receive data from the queue
        serial_number, color_image, depth_image, depth_colormap, rgb_timestamp, depth_timestamp, frame_number, fx, fy, ppx, ppy, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = queue.get()
        
        
        # Display or process frames as needed
        cv2.imshow(f"Color Stream - Camera {serial_number}", color_image)
        cv2.imshow(f"Depth Stream - Camera {serial_number}", depth_image)
        
        # Write metadata to CSV
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                frame_number, serial_number, rgb_timestamp, depth_timestamp, fx, fy, ppx, ppy,
                accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
            ])

        # Save RGB image
        rgb_filename = os.path.join(parent_folder, f"rgb_images_{date_string}", f"rgb_frame_{frame_number}_cam{serial_number}.png")
        cv2.imwrite(rgb_filename, color_image)
        
        # Save depth image
        depth_filename = os.path.join(parent_folder, f"depth_images_{date_string}", f"depth_frame_{frame_number}_cam{serial_number}.png")
        cv2.imwrite(depth_filename, np.asarray(depth_image).astype(np.uint16))
        
        # Save depth colormap
        depth_colormap_filename = os.path.join(parent_folder, f"depth_visuals_{date_string}", f"depth_colormap_{frame_number}_cam{serial_number}.png")
        cv2.imwrite(depth_colormap_filename, depth_colormap)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    freeze_support()
    
    args = parse_arguments()
    parent_folder = args.parent_folder
    
    # Get context for handling an arbitrary number of cameras
    ctx = rs.context()

    # Queue for sharing frames between processes
    frame_queue = Queue()

    date_string = datetime.today().strftime('%Y-%m-%d')
    
    # Create parent folder
    os.makedirs(parent_folder, exist_ok=True)

    # Configure settings for saving images
    rgb_folder = os.path.join(parent_folder, f"rgb_images_{date_string}")
    depth_folder = os.path.join(parent_folder, f"depth_images_{date_string}")
    depth_visuals_folder = os.path.join(parent_folder, f"depth_visuals_{date_string}")
    csv_filename = os.path.join(parent_folder, f"frame_metadata_{date_string}.csv")
    
    if os.path.exists(rgb_folder):
        shutil.rmtree(rgb_folder)
    if os.path.exists(depth_folder):
        shutil.rmtree(depth_folder)
    if os.path.exists(depth_visuals_folder):
        shutil.rmtree(depth_visuals_folder)
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(depth_visuals_folder, exist_ok=True)

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write CSV header
        writer.writerow(["frame_number", "serial_number", "rgb_timestamp",
                         "depth_timestamp", "fx", "fy", "ppx", "ppy",
                         "accel_x", "accel_y", "accel_z",
                         "gyro_x", "gyro_y", "gyro_z"])

    # Start a process for each camera
    processes = []
    num_cameras = len(ctx.devices)
    for device in ctx.devices:
        p = Process(target=capture_frames, args=(device.get_info(rs.camera_info.serial_number), frame_queue))
        processes.append(p)
        p.start()

    # Start a single process to handle frame processing
    process_frames_process = Process(target=process_frames, args=(frame_queue, csv_filename, date_string, num_cameras, parent_folder))
    process_frames_process.start()

    # Wait for all camera processes to finish
    for p in processes:
        p.join()

    # Terminate the frame processing process
    process_frames_process.terminate()
