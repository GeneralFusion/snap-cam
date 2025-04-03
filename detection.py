import os
import sys
import cv2
import csv
import yaml
import numpy as np

from typing import *

from image_detection_tool.image_process_helper.helper_functions import *


def classify_diode(image_path: str, calibration_file: str) -> Dict[str, Optional[bool]]:
    """
    Classify the diode status for each diode specified in the calibration file.

    Parameters:
        image_path (str): Path to the image file.
        calibration_file (str): Path to the YAML calibration file containing diode settings.

    Returns:
        dict: A dictionary mapping each diode (by its 'signal_map' identifier) to a boolean value indicating:
              - True if the diode is off (or the signal is as expected) 
              - False if the diode's signal is not meeting the threshold.
    """
    # Load calibration data from the YAML file.
    with open(calibration_file, 'r') as file:
        calibration_data = yaml.safe_load(file)
    calibration_data = calibration_data['CapCam1']

    # Load the image in grayscale mode.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to read image.")

    status: Dict[str, Optional[bool]] = {}

    # Iterate through each diode entry in the calibration file.
    for row_key, row_cams in calibration_data.items():
        for col_key, cam_dict in row_cams.items():
            x_pos = cam_dict['x_loc']
            y_pos = cam_dict['y_loc']
            capacitor_bank = cam_dict['id']
            diode_id = cam_dict['signal_map']
            connected = cam_dict['connected']

            # If diode is marked as disconnected or not applicable, we consider its status as "off"
            if connected == 'OFF' or capacitor_bank == 'NA':
                status[diode_id] = True
            else:
                # Calculate ROI boundaries for the diode.
                x_bounds, y_bounds = calculate_roi(x_center=x_pos, y_center=y_pos, delta=20)
                # Extract the ROI from the image.
                roi = image[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]
                # Evaluate the signal in the ROI.
                # Here, bright_value is set to 10 and the count threshold is half the total number of pixels in the ROI.
                signal_valid = evaluate_signal(roi_array=roi, bright_value=0, count_threshold=roi.size // 2)
                status[diode_id] = signal_valid

    return status


def main():
    status_list = classify_diode(image_path='pi3b_test_image/019000/CapCam1.png', calibration_file='capcam_cal/CapCam1.yml')
    print(status_list)
    mark_connected_diodes(image_path='pi3b_test_image/019000/CapCam1.png', calibration_file='capcam_cal/CapCam1.yml', status_array=status_list, output_path='output_image/test.png')

if __name__ == "__main__":
    main()
