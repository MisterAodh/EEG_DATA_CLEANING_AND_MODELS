import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat

# Load data from CSV and MAT files
vep_csv = pd.read_csv('../datasets/vep_data.csv')
vep_mat = loadmat('../datasets/vep.mat')['EEG']
stim_times = loadmat('../datasets/vep.mat')['stim_times'][0]

# Define parameters for dynamic plotting and analysis
window_size = 300
pre_samples = 100
post_samples = 150
threshold = 300

def dynamic_plot():
    """Handle dynamic plotting for the first 4 channels from the CSV data."""
    fig, axs = plt.subplots(2, 2)
    lines = []
    for i in range(4):  # We will use the first 4 channels
        sensor_data = vep_csv.iloc[:, i].values
        row, col = divmod(i, 2)
        axs[row, col].set_ylim([-1000, 1000])
        axs[row, col].axhline(y=0, color='red', linestyle='dotted', linewidth=1)
        line, = axs[row, col].plot(sensor_data[:window_size])
        lines.append(line)

    def update(frame):
        for idx, line in enumerate(lines):
            start = (frame * 5) % (len(vep_csv) - window_size)
            end = start + window_size
            line.set_ydata(vep_csv.iloc[start:end, idx].values)
        return lines

    ani = FuncAnimation(fig, update, frames=200, blit=True, interval=100)
    plt.show()

def epoch_data(eeg_data, channel_index, trig_times, pre_samples, post_samples, threshold):
    """Epoch the data and return good epochs."""
    epochs = []
    for trig_time in trig_times:
        start = trig_time - pre_samples
        end = trig_time + post_samples
        if 0 <= start and end < eeg_data.shape[1]:
            epoch = eeg_data[channel_index, start:end]
            if np.all(np.abs(epoch) <= threshold):
                epochs.append(epoch)
    return np.array(epochs)

def epoch_data_with_artifacts(eeg_data, channel_index, trig_times, pre_samples, post_samples, threshold):
    """Epoch the data, track and return samples with and without artifacts."""
    good_epochs = []
    artifact_epochs = []
    for trig_time in trig_times:
        start = trig_time - pre_samples
        end = trig_time + post_samples
        if 0 <= start and end < eeg_data.shape[1]:
            epoch = eeg_data[channel_index, start:end]
            if np.all(np.abs(epoch) <= threshold):
                good_epochs.append(epoch)
            else:
                artifact_epochs.append(epoch)
            # Collect up to 3 epochs for each type
            if len(good_epochs) == 3 and len(artifact_epochs) == 3:
                break
    print(f"Good epochs collected: {len(good_epochs)}")
    print(f"Artifact epochs collected: {len(artifact_epochs)}")
    return np.array(good_epochs), np.array(artifact_epochs)

def vep_analysis():
    """Perform VEP analysis for the first 9 channels and plot in a grid."""
    fig, axs = plt.subplots(3, 3)
    for i in range(9):
        epochs = epoch_data(vep_mat, i, stim_times, pre_samples, post_samples, threshold)
        if epochs.size > 0:
            average_signal = np.mean(epochs, axis=0)
        else:
            continue  # Skip plotting if no valid epochs
        row, col = divmod(i, 3)
        axs[row, col].plot(np.arange(-pre_samples, post_samples), average_signal)
        axs[row, col].set_title(f'Channel {i + 1}')
        axs[row, col].set_ylim([-150, 150])
        axs[row, col].axhline(y=0, color='red', linestyle='dotted', linewidth=1)
    plt.tight_layout()
    plt.show()

def plot_epochs(good_epochs, artifact_epochs):
    """Plot selected good and artifact epochs in a grid."""
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))  # 3x3 grid
    for i in range(3):
        if i < len(good_epochs):
            axs[i, 0].plot(good_epochs[i])
            axs[i, 0].set_title("Good Epoch {}".format(i+1))
            axs[i, 0].set_ylim([-160, 240])
        if i < len(artifact_epochs):
            axs[i, 1].plot(artifact_epochs[i])
            axs[i, 1].set_title("Artifact Epoch {}".format(i+1))
            axs[i, 1].set_ylim([-700, 400])
        axs[i, 2].axis('off')  # Empty plot for alignment
    plt.tight_layout()
    plt.show()

# Execute dynamic plotting
dynamic_plot()

input("Press any key to continue to VEP analysis...")

# Execute VEP analysis
vep_analysis()

input("Press any key to view epochs with and without artifacts...")

# Collect and display epochs with and without artifacts for one channel, e.g., channel 0
good_epochs, artifact_epochs = epoch_data_with_artifacts(vep_mat, 0, stim_times, pre_samples, post_samples, threshold)
# print(f"Good epochs: {good_epochs}")
# print(f"Artifact epochs: {artifact_epochs}")
plot_epochs(good_epochs, artifact_epochs)

input("Press any key to finish...")
print("Finished")
