import numpy as np
import pandas as pd
import json
import os

# Load data from CSV files
vep_csv = pd.read_csv('datasets/vep_data.csv')
stim_times_csv = pd.read_csv('datasets/vep_stim_times.csv')

# Define parameters for analysis
artifact_threshold = 300  # Threshold for artifact detection
artifact_window = 50  # Window size around artifact to set to 0
channel_index = 0  # Example channel to visualize

# Remove artifacts by zeroing out sections of data around high-value points
def remove_artifacts(data, threshold, window):
    for i in range(data.shape[0]):
        artifact_indices = np.where(np.abs(data[i]) > threshold)[0]
        for idx in artifact_indices:
            start = max(0, idx - window)
            end = min(len(data[i]), idx + window)
            data[i, start:end] = 0
    return data

# Detect VEP responses
def detect_vep_responses(data, threshold_high, threshold_low, search_window, response_window):
    detected_responses = []
    i = 0
    while i < len(data):
        if data[i] > threshold_high:
            end_window = min(i + search_window, len(data))
            peak_idx = np.argmax(data[i:end_window]) + i
            start_check = max(0, peak_idx - 50)
            end_check = peak_idx
            if np.any(data[start_check:end_check] < threshold_low):
                start_response = max(0, peak_idx - response_window)
                end_response = min(len(data), peak_idx + response_window)
                detected_responses.append((start_response, end_response))
                i = end_response
            else:
                i += 1
        else:
            i += 1
    return detected_responses

# Check if detected responses are true VEP responses and collect data periods
def check_true_veps_and_collect_periods(detected_responses, stim_times, data, tolerance=50):
    true_periods = []
    false_periods = []
    stim_times_detected = set()
    for start, end in detected_responses:
        is_true = False
        for stim in stim_times:
            if start <= stim <= end:
                is_true = True
                stim_times_detected.add(stim)
        if is_true:
            true_periods.append(data[start:end].tolist())
        else:
            false_periods.append(data[start:end].tolist())
    missed_stim_times = set(stim_times) - stim_times_detected
    return {
        "true_time_periods": true_periods,
        "false_time_periods": false_periods
    }, len(true_periods), len(false_periods), len(missed_stim_times)

# Main part of the code
if __name__ == "__main__":
    # Create directory for event dictionaries if it doesn't exist
    if not os.path.exists('event_dictionairies'):
        os.makedirs('event_dictionairies')

    # Preprocess the EEG data
    vep_mat_cleaned = remove_artifacts(vep_csv.values.T, artifact_threshold, artifact_window)
    print("Artifact removal by zeroing high-value sections completed.")

    stim_times = stim_times_csv['Stimulus_Times'].values

    for i in range(vep_mat_cleaned.shape[0]):  # Iterate over each channel
        # Detect VEP responses
        detected_responses = detect_vep_responses(vep_mat_cleaned[i], 30, -13, 100, 100)
        print(f"Detected {len(detected_responses)} VEP responses.")

        # Check if detected responses are true VEP responses and collect data periods
        response_dict, true_veps, false_veps, missed_stim_times = check_true_veps_and_collect_periods(detected_responses, stim_times, vep_mat_cleaned[i])
        print(f"Number of true VEP responses: {true_veps}")
        print(f"Number of false VEP responses: {false_veps}")
        print(f"Number of VEP time stamps not in a response: {missed_stim_times}")

        # Save the response dictionary to a JSON file
        with open(f'event_dictionairies/response_dict_channel_{i}.json', 'w') as f:
            json.dump(response_dict, f)
        print(f"Response dictionary saved to 'event_dictionairies/response_dict_channel_{i}.json'.")
# import json
# import os
#
#
# def process_json_file(file_path):
#     # Load the JSON file
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#
#     # Measure the length of the true and false time periods
#     true_count = len(data.get('true_time_periods', []))
#     false_count = len(data.get('false_time_periods', []))
#
#     # Add the counts to the data
#     data['true_count'] = true_count
#     data['false_count'] = false_count
#
#     # Save the updated JSON file
#     with open(file_path, 'w') as f:
#         json.dump(data, f, indent=4)
#
#
# # Directory containing the JSON files
# json_directory = 'event_dictionairies'
#
# # Process each JSON file in the directory
# for filename in os.listdir(json_directory):
#     if filename.endswith('.json'):
#         file_path = os.path.join(json_directory, filename)
#         process_json_file(file_path)
#
# print("Processing complete.")
