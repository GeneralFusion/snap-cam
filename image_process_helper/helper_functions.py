import os
import sys
import cv2
import csv
import yaml
import numpy as np

from typing import *

def count_blacked_out_rows(image_path):
    """Count how many rows in the image are completely black (sum of pixels = 0)."""
    # read image as grayscale so each pixel is a 0..255 intensity value
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read file: {image_path}")

    # Sum across each row; a sume of 0 means the entire row is blacked out
    row_sums = np.sum(image, axis=1)
    black_rows = np.where(row_sums == 0)[0]
    return len(black_rows)


def process_all_shots(root_dir, output_csv):
    """Helper function that take a repository for shot folders and analyze all images to produce a summary chart"""
    shot_results = []
    all_filenames = set()

    # First pass: gather all image filenames used across all folders
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path) or not folder.isdigit():
            continue
        for f in os.listdir(folder_path):
            if f.lower().endswith(".png"):
                all_filenames.add(f)

    # Sort filenames consistently
    sorted_filenames = sorted(all_filenames)

    # Second pass: evaluate each image in each shot folder
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path) or not folder.isdigit():
            continue

        row = [folder]
        for image_name in sorted_filenames:
            image_path = os.path.join(folder_path, image_name)
            if os.path.exists(image_path):
                blacked_out_count = count_blacked_out_rows(image_path)
                if blacked_out_count > 30:
                    row.append(1)
                else:
                    row.append(0)
            else:
                row.append("")  # File doesn't exist in this folder

        shot_results.append(row)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["shot"] + sorted_filenames
        writer.writerow(header)
        writer.writerows(shot_results)

    print(f"Image quality summary saved to {output_csv}")

def calculate_roi(x_center: int, y_center: int, delta: int = 20) -> Tuple[List[int], List[int]]:
    """
    Calculate the region of interest (ROI) boundaries based on the diode's center position.

    Parameters:
        x_center (int): X coordinate of the diode.
        y_center (int): Y coordinate of the diode.
        delta (int): Offset from the center to define the ROI.

    Returns:
        tuple: Two lists representing the x boundaries ([x_left, x_right]) and y boundaries ([y_lower, y_upper]).
    """
    x_left  = max(x_center - delta, 0)
    y_lower = max(y_center - delta, 0)
    x_right = x_center + delta
    y_upper = y_center + delta
    return [x_left, x_right], [y_lower, y_upper]

def evaluate_signal(roi_array: np.ndarray, bright_value: int, count_threshold: int) -> bool:
    """
    Determine if the diode signal is active in the given ROI.

    Parameters:
        roi_array (np.ndarray): The region of interest extracted from the image.
        bright_value (int): The pixel intensity threshold to consider "bright."
        count_threshold (int): The minimum number of pixels with intensity > bright_value
                               required to consider the diode "on."

    Returns:
        bool: True if bright pixels >= count_threshold; otherwise, False.
    """
    # If ROI is uniformly a single intensity, it's likely not lit
    unique_values, counts = np.unique(roi_array, return_counts=True)
    if len(unique_values) < 2:
        return False
    
    # Count *all* pixels above the bright_value
    mask = unique_values > bright_value
    if not np.any(mask):
        # No pixel intensities above bright_value
        return False
    
    # Sum up counts of all intensities greater than bright_value
    total_bright_pixels = counts[mask].sum()
    return total_bright_pixels >= count_threshold

def mark_connected_diodes(image_path: str, calibration_file: str, status_array: List[str], output_path: str, marker_size: int = 2, thickness: int = 1) -> None:
    """
    Load an image and calibration file, mark a red "X" at the (x, y) position for each diode 
    with 'connected' equal to "ON", and save the new image.
    
    Parameters:
        image_path (str): Path to the input image.
        calibration_file (str): Path to the YAML calibration file.
        output_path (str): Path where the marked image will be saved.
        marker_size (int): Size of the marker.
        thickness (int): Thickness of the marker lines.
    """
    # Load the image in color mode
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or unable to read image at: " + image_path)
    
    # Load calibration data from the YAML file
    with open(calibration_file, 'r') as file:
        calibration_data = yaml.safe_load(file)
    
    calibration_data = calibration_data['CapCam1']
    # Iterate through the calibration data
    for row_key, row_data in calibration_data.items():
        for col_key, cam_dict in row_data.items():
            # Only mark diodes with connected = "ON"
            if cam_dict['connected'] == "ON":
                x_pos = int(cam_dict['x_loc'])
                y_pos = int(cam_dict['y_loc'])
                
                # Draw a blue "X" using two crossing lines
                color = (255, 0, 0)  # Blue in BGR
                pt1 = (x_pos - marker_size, y_pos - marker_size)
                pt2 = (x_pos + marker_size, y_pos + marker_size)
                pt3 = (x_pos - marker_size, y_pos + marker_size)
                pt4 = (x_pos + marker_size, y_pos - marker_size)
                
                cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
                cv2.line(image, pt3, pt4, color, thickness, lineType=cv2.LINE_AA)

        
                if not status_array[cam_dict['signal_map']]:
          
                    y_pos += 5
                    # Draw a blue "X" using two crossing lines
                    color = (0, 0, 255)
                    pt1 = (x_pos - marker_size, y_pos - marker_size)
                    pt2 = (x_pos + marker_size, y_pos + marker_size)
                    pt3 = (x_pos - marker_size, y_pos + marker_size)
                    pt4 = (x_pos + marker_size, y_pos - marker_size)
                    cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
                    cv2.line(image, pt3, pt4, color, thickness, lineType=cv2.LINE_AA)

    
    # Save the modified image to the output path
    cv2.imwrite(output_path, image)

