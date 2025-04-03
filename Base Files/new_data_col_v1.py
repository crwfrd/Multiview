# the visualization was fixed

import pyrealsense2 as rs
from multiprocessing import Process, Queue, freeze_support
import cv2
import numpy as np
import csv
import os
import argparse
import shutil
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-camera recorder with IMU data")
    parser.add_argument("parent_folder", type=str, help="Parent folder name for storing data")
    return parser.parse_args()


def capture_frames(serial_number, queue):
    """
    Capture frames from a single RealSense camera (identified by serial_number),
    and put them (and associated metadata) onto a multiprocessing queue.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Lock to a specific device by serial number
    config.enable_device(serial_number)

    # Enable color, depth, and IMU streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    # Start streaming
    profile = pipeline.start(config)

    # Depth scale (to convert depth pixels to meters), if needed
    sensor = profile.get_device().first_depth_sensor()
    depth_scale = sensor.get_depth_scale()  # e.g. ~0.001 for D435

    # Optional filters (disabled by default)
    # spatial_filter = rs.spatial_filter()
    # temporal_filter = rs.temporal_filter()

    frame_number = 0

    try:
        while True:
            # Wait for a coherent frameset (color, depth, possibly IMU)
            frames = pipeline.wait_for_frames()

            if not frames:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            # If any required frames are missing, skip
            if not color_frame or not depth_frame or not accel_frame or not gyro_frame:
                continue

            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # If you want filtering, you could do something like:
            # filtered_depth = spatial_filter.process(depth_frame)
            # filtered_depth = temporal_filter.process(filtered_depth)
            # depth_image = np.asanyarray(filtered_depth.get_data())

            # Intrinsics (optional if you don't need them in CSV)
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            fx, fy, ppx, ppy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

            # IMU data
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            accel_x, accel_y, accel_z = accel_data.x, accel_data.y, accel_data.z
            gyro_x, gyro_y, gyro_z = gyro_data.x, gyro_data.y, gyro_data.z

            # Timestamps
            rgb_timestamp = color_frame.get_timestamp()
            depth_timestamp = depth_frame.get_timestamp()

            # ------------------------------------------
            # Basic Depth Visualization (Fixed Range)
            # ------------------------------------------
            # 1) Convert depth to meters for consistent range.
            #    e.g. if depth_scale ~ 0.001, then depth_in_meters = depth_image * 0.001
            depth_in_meters = depth_image * depth_scale

            # 2) Choose a min/max range in meters for display. 
            #    Example: 0.2 m to 3.0 m. Adjust as needed for your use case.
            min_m = 0.2
            max_m = 3.0

            # 3) Clamp values and scale to [0..255].
            depth_clamped = np.clip(depth_in_meters, min_m, max_m)
            # Map min_m => 0, max_m => 255
            depth_scaled = 255 * (depth_clamped - min_m) / (max_m - min_m)
            depth_scaled = depth_scaled.astype(np.uint8)

            # 4) Invert so closer => warm, farther => cool
            depth_inverted = 255 - depth_scaled

            # 5) Apply color map
            depth_colormap = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)

            frame_number += 1

            # Put data in the queue
            queue.put((
                serial_number, 
                color_image, 
                depth_image, 
                depth_colormap,
                rgb_timestamp, 
                depth_timestamp, 
                frame_number,
                fx, fy, ppx, ppy, 
                accel_x, accel_y, accel_z, 
                gyro_x, gyro_y, gyro_z
            ))

    finally:
        pipeline.stop()


def process_frames(queue, csv_filename, date_string, num_cameras, parent_folder):
    while True:
        # Receive data from the queue
        (serial_number,
         color_image,
         depth_image,
         depth_colormap,
         rgb_timestamp,
         depth_timestamp,
         frame_number,
         fx, fy, ppx, ppy,
         accel_x, accel_y, accel_z,
         gyro_x, gyro_y, gyro_z) = queue.get()
        
        # -----------------------
        # 1) Create / confirm subfolders per camera
        camera_folder = os.path.join(parent_folder, serial_number)
        rgb_folder = os.path.join(camera_folder, f"rgb_images_{date_string}")
        depth_folder = os.path.join(camera_folder, f"depth_images_{date_string}")
        depth_visuals_folder = os.path.join(camera_folder, f"depth_visuals_{date_string}")
        
        os.makedirs(rgb_folder, exist_ok=True)
        os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(depth_visuals_folder, exist_ok=True)
        # -----------------------
        
        # Display or process frames as needed
        cv2.imshow(f"Color Stream - Camera {serial_number}", color_image)
        cv2.imshow(f"Depth Stream - Camera {serial_number}", depth_colormap)
        
        # Write metadata to CSV (shared among all cameras, if that's what you want)
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                frame_number, serial_number,
                rgb_timestamp, depth_timestamp,
                fx, fy, ppx, ppy,
                accel_x, accel_y, accel_z,
                gyro_x, gyro_y, gyro_z
            ])

        # Save RGB image
        rgb_filename = os.path.join(rgb_folder,
            f"rgb_frame_{frame_number}_fx_{fx}_fy_{fy}_ppx_{ppx}_ppy_{ppy}_accelx_{accel_x}__accely_{accel_y}_accel_z_{accel_z}_gyrox_{gyro_x}_gyroy_{gyro_y}_gyroz_{gyro_z}.png")
        cv2.imwrite(rgb_filename, color_image)
        
        # Save depth image (as 16-bit)
        depth_filename = os.path.join(depth_folder,
            f"depth_frame_{frame_number}_fx_{fx}_fy_{fy}_ppx_{ppx}_ppy_{ppy}_accelx_{accel_x}__accely_{accel_y}_accel_z_{accel_z}_gyrox_{gyro_x}_gyroy_{gyro_y}_gyroz_{gyro_z}.png")
        cv2.imwrite(depth_filename, depth_image.astype(np.uint16))
        
        # Save depth colormap
        depth_colormap_filename = os.path.join(depth_visuals_folder,
            f"depth_colormap_{frame_number}_fx_{fx}_fy_{fy}_ppx_{ppx}_ppy_{ppy}_accelx_{accel_x}__accely_{accel_y}_accel_z_{accel_z}_gyrox_{gyro_x}_gyroy_{gyro_y}_gyroz_{gyro_z}.png")
        cv2.imwrite(depth_colormap_filename, depth_colormap)
        
        # Check for user quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    freeze_support()

    args = parse_arguments()
    parent_folder = args.parent_folder

    # Create a RealSense context
    ctx = rs.context()

    # Multiprocessing queue for frames
    frame_queue = Queue()

    # Get today's date as a string
    date_string = datetime.today().strftime('%Y-%m-%d')

    # Make sure the parent folder exists
    os.makedirs(parent_folder, exist_ok=True)

    # Build sub-folders
    # rgb_folder = os.path.join(parent_folder, f"rgb_images_{date_string}")
    # depth_folder = os.path.join(parent_folder, f"depth_images_{date_string}")
    # depth_visuals_folder = os.path.join(parent_folder, f"depth_visuals_{date_string}")
    csv_filename = os.path.join(parent_folder, f"frame_metadata_{date_string}.csv")

    # Remove existing folders/files (if you want a clean start)
    # for folder in [rgb_folder, depth_folder, depth_visuals_folder]:
    #     if os.path.exists(folder):
    #         shutil.rmtree(folder)
    #     os.makedirs(folder, exist_ok=True)

    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    # Create CSV header
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame_number", "serial_number",
            "rgb_timestamp", "depth_timestamp",
            "fx", "fy", "ppx", "ppy",
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z"
        ])

    # Start a capture process for each detected RealSense device
    processes = []
    num_cameras = len(ctx.devices)
    for device in ctx.devices:
        serial = device.get_info(rs.camera_info.serial_number)
        p = Process(target=capture_frames, args=(serial, frame_queue))
        processes.append(p)
        p.start()

    # Start a single process to handle frame saving and CSV logging
    process_frames_process = Process(
        target=process_frames,
        args=(frame_queue, csv_filename, date_string, num_cameras, parent_folder)
    )
    process_frames_process.start()

    # Wait for capture processes to finish (usually this is indefinite unless you kill them)
    for p in processes:
        p.join()

    # After capture processes end, stop the processing as well
    process_frames_process.terminate()
