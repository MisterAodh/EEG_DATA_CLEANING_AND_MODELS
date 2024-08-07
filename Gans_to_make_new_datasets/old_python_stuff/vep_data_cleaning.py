import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Load data from CSV files
vep_csv = pd.read_csv('../datasets/vep_data.csv')
stim_times_csv = pd.read_csv('../datasets/vep_stim_times.csv')
stim_times = stim_times_csv[stim_times_csv.columns[0]].values

# Define parameters for analysis
sampling_rate = 512  # Hz
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

# Epoching function
def epoch_data(eeg_data, trig_times, pre_samples=100, post_samples=150):
    epochs = []
    for trig_time in trig_times:
        start = trig_time - pre_samples
        end = trig_time + post_samples
        if 0 <= start and end < eeg_data.shape[1]:
            epoch = eeg_data[:, start:end]
            epochs.append(epoch)
    return np.array(epochs)

# Plot ICA components function
def plot_ica_components(sources, n_components, frequency_bands):
    plt.figure(figsize=(12, 12))
    for i in range(n_components):  # Plot components
        plt.subplot(n_components, 1, i + 1)
        plt.plot(sources[i], label=f'ICA Component {i + 1} - {frequency_bands[i % len(frequency_bands)]}')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Plot average VEP for each component
def plot_average_vep(ica_epochs, n_components, frequency_bands, pre_samples, post_samples):
    fig, axs = plt.subplots(n_components, 1, figsize=(12, 15))
    time_axis = np.linspace(-pre_samples, post_samples, pre_samples + post_samples)
    for i in range(n_components):
        average_vep = np.mean(ica_epochs[:, i, :], axis=0)
        axs[i].plot(time_axis, average_vep, label=f'ICA Component {i + 1} - {frequency_bands[i % len(frequency_bands)]}')
        axs[i].set_title(f'Average VEP - ICA Component {i + 1}')
        axs[i].set_xlabel('Time (samples)')
        axs[i].set_ylabel('Amplitude (uV)')
        axs[i].legend()
    plt.tight_layout()
    plt.show()

# Plot individual VEP events
def plot_individual_veps(epochs, title, num_events, pre_samples, post_samples, channel_index):
    fig, axs = plt.subplots(num_events, 1, figsize=(12, 12))
    time_axis = np.linspace(-pre_samples, post_samples, pre_samples + post_samples)
    for i in range(num_events):
        axs[i].plot(time_axis, epochs[i, channel_index, :], label=f'Event {i + 1}')
        axs[i].set_title(f'{title} - Event {i + 1}')
        axs[i].set_xlabel('Time (samples)')
        axs[i].set_ylabel('Amplitude (uV)')
        axs[i].legend()
    plt.tight_layout()
    plt.show()

# Visualize the cleaned data vs. the filtered data
def visualize_cleaned_vs_filtered(vep_csv, vep_mat_cleaned, cleaned_eeg, channel_index):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(vep_csv.values[:, channel_index], label='Raw Data')
    plt.title('Raw EEG Data - Channel {}'.format(channel_index + 1))
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(vep_mat_cleaned[channel_index], label='Cleaned Data', color='orange')
    plt.title('Cleaned EEG Data - Channel {}'.format(channel_index + 1))
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(cleaned_eeg[channel_index], label='Reconstructed Data', color='green')
    plt.title('Reconstructed EEG Data - Channel {}'.format(channel_index + 1))
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main part of the code
if __name__ == "__main__":
    # Preprocess the EEG data
    vep_mat_cleaned = remove_artifacts(vep_csv.values.T, artifact_threshold, artifact_window)
    print("Artifact removal by zeroing high-value sections completed.")

    # Apply ICA to the EEG data
    def apply_ica(eeg_data, n_components):
        ica = FastICA(n_components=n_components, random_state=0)
        sources = ica.fit_transform(eeg_data.T).T  # Transpose to match the expected shape for ICA
        return ica, sources

    # Get user input for the number of ICA components
    n_components = int(input("Enter the number of ICA components: "))

    # Apply ICA with user-defined number of components
    ica, sources = apply_ica(vep_mat_cleaned, n_components)
    print("ICA completed with {} components.".format(n_components))

    # Define frequency bands for labeling
    frequency_bands = ["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-12 Hz)", "Beta (12-30 Hz)", "Gamma (>30 Hz)"]

    # Plot ICA components
    plot_ica_components(sources, n_components, frequency_bands)

    input("Press any key to continue to VEP epoch extraction...")

    # Extract VEP epochs for each component
    pre_samples = 100
    post_samples = 150
    ica_epochs = epoch_data(sources, stim_times, pre_samples, post_samples)

    # Plot average VEP for each component
    plot_average_vep(ica_epochs, n_components, frequency_bands, pre_samples, post_samples)

    # User input to remove components
    components_to_remove = []
    while True:
        to_remove = input("What do you want to remove (1-{}, or 'zzz' to finish): ".format(n_components))
        if to_remove.lower() == 'zzz':
            break
        elif to_remove.isdigit() and 1 <= int(to_remove) <= n_components:
            components_to_remove.append(int(to_remove) - 1)

    # Remove selected components
    clean_sources = sources.copy()
    clean_sources[components_to_remove, :] = 0

    # Reconstruct the cleaned signal without the selected components
    cleaned_eeg = ica.inverse_transform(clean_sources.T).T
    print("Selected ICA components removed and signal reconstructed.")

    # Save the cleaned EEG data to a CSV file
    cleaned_eeg_df = pd.DataFrame(cleaned_eeg.T, columns=[f'Sensor{i+1}' for i in range(cleaned_eeg.shape[0])])
    cleaned_eeg_df.to_csv('datasets/cleaned_vep_data.csv', index=False)
    print("Cleaned EEG data saved to 'datasets/cleaned_vep_data.csv'.")

    # Visualize the cleaned data vs. the filtered data
    visualize_cleaned_vs_filtered(vep_csv, vep_mat_cleaned, cleaned_eeg, channel_index)

    input("Press any key to continue to average VEP epoch extraction for cleaned data...")

    # Extract VEP epochs for the cleaned data
    cleaned_ica_epochs = epoch_data(cleaned_eeg, stim_times, pre_samples, post_samples)

    # Calculate and plot the average VEP for the cleaned data
    fig, ax = plt.subplots(figsize=(12, 6))
    time_axis = np.linspace(-pre_samples, post_samples, pre_samples + post_samples)
    average_cleaned_vep = np.mean(cleaned_ica_epochs, axis=0)
    average_cleaned_vep = np.mean(average_cleaned_vep, axis=0)  # Ensure correct averaging
    ax.plot(time_axis, average_cleaned_vep, label='Average VEP - Cleaned Data')
    ax.set_title('Average VEP - Cleaned Data')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude (uV)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("Average VEP for cleaned data plotted.")

    # Plot individual VEP events before and after cleaning
    num_events_to_plot = 5  # Number of events to plot

    # Plot individual VEP events before cleaning
    plot_individual_veps(ica_epochs, 'Individual VEP Before Cleaning', num_events_to_plot, pre_samples, post_samples, channel_index)

    # Plot individual VEP events after cleaning
    plot_individual_veps(cleaned_ica_epochs, 'Individual VEP After Cleaning', num_events_to_plot, pre_samples, post_samples, channel_index)

    print("Individual VEP events plotted before and after cleaning.")
