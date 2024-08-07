# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import loadmat
#
# def calculate_snr(signal, noise):
#     signal_rms = np.sqrt(np.mean(np.square(signal)))
#     noise_rms = np.sqrt(np.mean(np.square(noise)))    # Calculate RMS of signal and noise
#     signal_power = np.square(signal_rms)
#     noise_power = np.square(noise_rms)    # Calculate powers
#     SNR = 10 * np.log10(signal_power / noise_power)    # Calculate SNR in decibels
#     return SNR
#
# # Epoching function
#
# def epoch_data(eeg_data, trig_times, triggers=None, pre_samples=0,
#                post_samples=0, channel_index=None, trigger_values=None):
#     epochs = []
#     for i, trig_time in enumerate(trig_times):
#         epoch_start = trig_time - pre_samples
#         epoch_end = trig_time + post_samples
#         if 0 <= epoch_start < eeg_data.shape[1] and epoch_end <= eeg_data.shape[1]:
#             if channel_index is not None:  # For VEP
#                 epoch = eeg_data[channel_index, epoch_start:epoch_end]
#                 if np.all(np.abs(epoch) <= 300):
#                     # filtering outliers for VEP I know that these
#                     # are just artifacts but i find the removal of outliers abhorant
#                     epochs.append(epoch)
#             else:  # For P300
#                 epoch = eeg_data[0, epoch_start:epoch_end]
#                 if triggers is not None and triggers[i] in trigger_values:
#                     epochs.append(epoch)
#     return np.array(epochs)
# # Load datasets
# p300_data = loadmat('P300.mat')
# vep_data = loadmat('vep.mat')
#
# # base variables
# sampling_rate = 512  # Hz for P300
# pre_stimulus_ms_p300 = 100  # before stimulus for P300
# post_stimulus_ms_p300 = 500  # after stimulus for P300
# pre_stimulus_samples_p300 = int(pre_stimulus_ms_p300 * sampling_rate / 1000)
# post_stimulus_samples_p300 = int(post_stimulus_ms_p300 * sampling_rate / 1000)
# id_number = 5
# pre_stimulus_samples_vep = 100  # before stimulus for VEP
# post_stimulus_samples_vep = 200  # after stimulus for VEP
#
# # P300 epoching
# epochs_standard = epoch_data(p300_data['eeg'],
#                              p300_data['trig_times'][0], p300_data['trigs'][0],
#                              pre_stimulus_samples_p300, post_stimulus_samples_p300,
#                              trigger_values=[1])
# epochs_deviant = epoch_data(p300_data['eeg'],
#                             p300_data['trig_times'][0], p300_data['trigs'][0],
#                             pre_stimulus_samples_p300, post_stimulus_samples_p300,
#                             trigger_values=[2])
#
# # VEP epoching
# # python uses zero indexing so gotta set the channel
# # back one as the channel for my id number would be different on matlab.
#
# epochs_vep = epoch_data(vep_data['EEG'], vep_data['stim_times'][0],
#                         pre_samples=pre_stimulus_samples_vep,
#                         post_samples=post_stimulus_samples_vep,
#                         channel_index=id_number - 1)
# # Calculate averages
# average_epoch_standard = np.mean(epochs_standard, axis=0)
# average_epoch_deviant = np.mean(epochs_deviant, axis=0)
# average_epoch_vep = np.mean(epochs_vep, axis=0)
#
# # Time axis for plotting
# time_axis_p300 = np.linspace(-pre_stimulus_ms_p300, post_stimulus_ms_p300, len(average_epoch_standard))
# time_axis_vep = np.arange(-pre_stimulus_samples_vep, post_stimulus_samples_vep)
# signal_vep = average_epoch_vep[pre_stimulus_samples_vep + 1:] #Getting the data ready to calculate SNR
# noise_vep = average_epoch_vep[:pre_stimulus_samples_vep]
# snr_vep = calculate_snr(signal_vep, noise_vep)
#
# # Save SNR to a text file
# results_file_path = 'findings/results.txt'  # path is discussed at end
# with open(results_file_path, 'w') as results_file:
#     results_file.write(f'SNR of the VEP response in dB: {snr_vep:.2f}\n')
#
# # Plotting P300 and  saving to findings
# plt.figure()
# plt.plot(time_axis_p300, average_epoch_standard, label='Standard', color='blue')
# plt.plot(time_axis_p300, average_epoch_deviant, label='Deviant', color='red')
# plt.xlabel('Time(measurements)')
# plt.ylabel('Voltage (µV)')
# plt.legend()
# plt.title('Average ERP Responses: standard vs Deviant- P300')
# plt.savefig('findings/average_erp_responses_p300.png')  # pathing is discussed later
#
# # Plot VEP responses  and saving to findings
# plt.figure()
# plt.plot(time_axis_vep, average_epoch_vep, label=f'VEP Channel {id_number}')
# plt.xlabel('Time(measurements)')
# plt.ylabel('Voltage (µV)')
# plt.legend()
# plt.title('Average VEP Response')
# plt.savefig('findings/average_vep_response.png')
#
# # Note on the plots if you run this code. So in my current working directory (cwd)
# # I have a folder called findings
# # which I save my plots to but if you dont have this folderyou will get an error.
# # In some IDE's (cwd) is funny but in pycharm if you allow the IDE to create the
# # env then the cwd will just be where your main.py file is
#
#
# import numpy as np



# import pandas as pd
# from scipy.io import loadmat
#
# # Load VEP data from the .mat file
# mat_contents = loadmat('vep.mat')
# vep_data = mat_contents['EEG']
#
# # Check the shape of the data to confirm correct loading
# print("Shape of VEP data:", vep_data.shape)
#
# # Assuming the data is organized as channels (10) x samples (65536)
# # Transpose the data to make each row correspond to a sample and each column to a sensor
# vep_data_transposed = vep_data.T
#
# # Create a DataFrame with appropriate column labels
# column_labels = [f'Sensor{i+1}' for i in range(vep_data.shape[0])]
# vep_df = pd.DataFrame(vep_data_transposed, columns=column_labels)
#
# # Save the DataFrame to a CSV file
# csv_path = 'vep_data.csv'
# vep_df.to_csv(csv_path, index=False)
# print(f"VEP data has been saved to '{csv_path}'.")
# import numpy as np
# import pandas as pd
# from scipy.io import loadmat
#
# # Load VEP data from the .mat file
# mat_contents = loadmat('vep.mat')
# vep_data = mat_contents['EEG']
#
# # Transpose the data if necessary
# vep_data_transposed = vep_data.T
#
# # Create a DataFrame with appropriate column labels
# column_labels = [f'Sensor{i+1}' for i in range(vep_data.shape[0])]
# vep_df = pd.DataFrame(vep_data_transposed, columns=column_labels)
#
# # Round all values in the DataFrame to six decimal places
# vep_df = vep_df.round(6)
#
# # Save the DataFrame to a CSV file
# csv_path = 'vep_data.csv'
# vep_df.to_csv(csv_path, index=False)
# print(f"VEP data has been saved to '{csv_path}' with values rounded to 6 decimal places.")









import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# Load data from CSV file
vep_df = pd.read_csv('vep_data.csv')

# Assuming you want to use the first sensor's data for dynamic plotting
# Convert to numpy array for easier manipulation
eeg_data = vep_df['Sensor1'].values

# Parameters for dynamic plotting
window_size = 300  # Number of points to display at a time
update_size = 5  # Number of points to update each frame
data_length = len(eeg_data)  # Get the length of the EEG data

# Set up the figure and axis for plotting
fig, ax = plt.subplots()
x = np.arange(window_size)  # x-axis data points
line, = ax.plot(x, eeg_data[:window_size])  # Initial plot slice
ax.set_ylim(-1000, 1000)  # Set fixed y-axis limits

# Add a red dotted line at y=0
ax.axhline(y=0, color='red', linestyle='dotted', linewidth=1)

def update(frame):
    start = (frame * update_size) % (data_length - window_size)  # Calculate the starting index
    end = start + window_size  # Calculate the ending index
    new_data = eeg_data[start:end]
    line.set_ydata(new_data)  # Update the data shown on the plot
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=range(data_length // update_size), blit=True, interval=100)  # interval in milliseconds

plt.title('Dynamic EEG Data Plot of Sensor1')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()
# import scipy.io
# import pandas as pd
#
# # Load the MAT file
# mat_contents = scipy.io.loadmat('vep.mat')
#
# # Extract stim_times from the loaded data
# stim_times = mat_contents['stim_times']  # Adjust indexing if needed based on the structure
# stim_times = stim_times[0]  # Adjust indexing if needed based on the structure
#
# # Create a DataFrame with appropriate column labels
# stim_times_df = pd.DataFrame(stim_times, columns=['Stimulus_Times'])
#
# # Save the DataFrame to a CSV file
# stim_times_df.to_csv('vep_stim_times.csv', index=False)
#
# print("Stimulus times have been saved to 'vep_stim_times.csv'.")
