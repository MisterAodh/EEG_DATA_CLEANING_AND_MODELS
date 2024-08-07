import numpy as np
import matplotlib.pyplot as plt
import json

# Load the error JSON file
with open('errors.json', 'r') as f:
    errors = json.load(f)

false_positives = np.array(errors['false_positives'])
false_negatives = np.array(errors['false_negatives'])

# Function to plot multiple signals
def plot_signals(signals, title, num_signals=9):
    plt.figure(figsize=(15, 15))
    for i in range(num_signals):
        plt.subplot(3, 3, i + 1)
        plt.plot(signals[i])
        plt.title(f'{title} {i + 1}')
    plt.tight_layout()
    plt.show()

# Plot 9 false positives
plot_signals(false_positives, "False Positive", num_signals=9)

# Plot 9 false negatives
plot_signals(false_negatives, "False Negative", num_signals=9)

# Function to plot the average of signals
def plot_average_signal(signals, title):
    avg_signal = np.mean(signals, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_signal)
    plt.title(f'Average {title}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()

# Plot average of false positives
plot_average_signal(false_positives, "False Positive")

# Plot average of false negatives
plot_average_signal(false_negatives, "False Negative")
