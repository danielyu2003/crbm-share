import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from crbm import CRBM
from pathlib import Path
from preprocess import extract, lowPassFilter

np.random.seed(21)

# Load and preprocess data
df = pd.read_csv("Apple_016.csv", skiprows=4, low_memory=False)
TORSO = '771BCC09528711EEC6A1D77FF14871BB'
quaternions = extract(df, TORSO)[:, 0]
# quaternions = extract(df, TORSO)
quaternions = lowPassFilter(quaternions)
# Downsample by 10
quaternions = quaternions[::10]

steps = len(quaternions)

# Initialize CRBM parameters
n_visible = 50
n_cond = 200
n_hidden = 600
numInds = 100
n_epochs = 3000

# Generate random indices using Numpy
randInds = np.random.randint(n_cond, steps - n_visible, size=numInds)

# Initialize CRBM and reconstruction error list
motionCrbm = CRBM(n_visible, n_hidden, n_cond)
reconstruction_errors = np.zeros((n_epochs, numInds))

# Precompute data slices for efficiency
v_data_list = np.array([quaternions[i:i + n_visible] for i in randInds])
cond_data_list = np.array([quaternions[i - n_cond:i] for i in randInds])

weightPath = Path("weights/W.npy")

if not weightPath.exists():
    # Training loop
    for epoch in range(n_epochs):
        for i, (v_data, cond_data) in enumerate(zip(v_data_list, cond_data_list)):
            v_data = v_data.reshape(1, -1)
            cond_data = cond_data.reshape(1, -1)

            # Perform contrastive divergence
            motionCrbm.contrastive_divergence(v_data, cond_data, lr=0.001)

            # Compute reconstruction error
            v_reconstructed = motionCrbm.reconstruct(v_data, cond_data)
            mse = np.mean((v_data - v_reconstructed) ** 2)
            reconstruction_errors[epoch, i] = mse

        # Print mean reconstruction error for the epoch
        epoch_mse = reconstruction_errors[epoch].mean()
        print(f"Epoch {epoch + 1}/{n_epochs}, Reconstruction Error: {epoch_mse:.4f}")

    motionCrbm.save_weights()
    # Plot reconstruction errors over epochs
    plt.plot(reconstruction_errors.mean(axis=1))
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Reconstruction Error")
    plt.show()

else:
    motionCrbm.load_weights()
    # Test and visualize reconstruction
    test_visible = quaternions[randInds[0]:randInds[0] + n_visible]
    test_cond = quaternions[randInds[0] - n_cond:randInds[0]]

    v_reconstructed = motionCrbm.reconstruct(test_visible, test_cond)
    v_reconstructed = lowPassFilter(v_reconstructed)
    v_reconstructed = lowPassFilter(v_reconstructed)
    v_reconstructed = lowPassFilter(v_reconstructed)

    plt.plot(test_visible, label="Original (First Feature)", c='r', alpha=0.7)
    plt.plot(v_reconstructed, label="Reconstructed (First Feature)", c='b', alpha=0.7)
    plt.legend()
    plt.grid(True)
    plt.show()
