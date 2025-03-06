import os
import numpy as np
from scipy.special import expit


class CRBM:
    def __init__(self, n_visible, n_hidden, n_cond):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_cond = n_cond

        # Initialize weights with small random values
        self.W = np.random.normal(0, 0.001, size=(n_visible, n_hidden))
        self.U = np.random.normal(0, 0.001, size=(n_cond, n_hidden))
        self.V = np.random.normal(0, 0.001, size=(n_cond, n_visible))

        # Initialize biases
        self.b = np.zeros(n_visible)  # Visible bias
        self.c = np.zeros(n_hidden)  # Hidden bias

    def sample_hidden(self, v, cond):
        """Sample hidden units from the visible and conditional inputs."""
        pre_activation = np.dot(v, self.W) + np.dot(cond, self.U) + self.c
        h_probs = expit(pre_activation)  # Sigmoid activation
        h_sample = np.random.binomial(1, h_probs)
        return h_sample, h_probs

    def sample_visible(self, h, cond):
        """Sample visible units given hidden and conditional inputs."""
        v_mean = np.dot(h, self.W.T) + np.dot(cond, self.V) + self.b
        v_sample = v_mean + np.random.normal(0, 0.001, size=v_mean.shape)
        return v_sample

    def contrastive_divergence(self, v_input, cond, k=1, lr=0.01):
        """Perform Contrastive Divergence with k Gibbs sampling steps."""
        # Positive phase
        h_sample, h_probs = self.sample_hidden(v_input, cond)
        positive_grad_W = np.dot(v_input.T, h_probs)
        positive_grad_U = np.dot(cond.T, h_probs)

        # Negative phase
        v_model = v_input.copy()
        for _ in range(k):
            h_model, _ = self.sample_hidden(v_model, cond)
            v_model = self.sample_visible(h_model, cond)

        h_model_probs = expit(np.dot(v_model, self.W) + np.dot(cond, self.U) + self.c)
        negative_grad_W = np.dot(v_model.T, h_model_probs)
        negative_grad_U = np.dot(cond.T, h_model_probs)
        negative_grad_V = np.dot(cond.T, (v_input - v_model))

        # Compute gradients
        dW = positive_grad_W - negative_grad_W
        dU = positive_grad_U - negative_grad_U
        dV = negative_grad_V
        db = np.mean(v_input - v_model, axis=0)
        dc = np.mean(h_probs - h_model_probs, axis=0)

        # Update weights and biases
        self.W += lr * dW
        self.U += lr * dU
        self.V += lr * dV
        self.b += lr * db
        self.c += lr * dc

    def train(self, v_data, cond_data, n_epochs=100, lr=0.01, batch_size=32):
        """Train the CRBM with given visible and conditional data."""
        n_samples = v_data.shape[0]
        reconstruction_errors = []

        for epoch in range(n_epochs):
            for i in range(0, n_samples, batch_size):
                batch_v = v_data[i:i + batch_size]
                batch_cond = cond_data[i:i + batch_size]
                self.contrastive_divergence(batch_v, batch_cond, k=1, lr=lr)

            # Compute reconstruction error for monitoring
            v_reconstructed = self.reconstruct(v_data, cond_data)
            mse = np.mean((v_data - v_reconstructed) ** 2)
            reconstruction_errors.append(mse)
            print(f"Epoch {epoch + 1}/{n_epochs}, Reconstruction Error: {mse:.4f}")

        return reconstruction_errors

    def reconstruct(self, v, cond):
        """Reconstruct visible units given conditional inputs."""
        h_sample, _ = self.sample_hidden(v, cond)
        v_recon = self.sample_visible(h_sample, cond)
        return v_recon

    def save_weights(self, directory="weights"):
        """Save model weights and biases to .npy files."""
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create directory if it doesn't exist

        np.save(os.path.join(directory, "W.npy"), self.W)
        np.save(os.path.join(directory, "U.npy"), self.U)
        np.save(os.path.join(directory, "V.npy"), self.V)
        np.save(os.path.join(directory, "b.npy"), self.b)
        np.save(os.path.join(directory, "c.npy"), self.c)

        print(f"Weights saved to '{directory}/'")

    def load_weights(self, directory="weights"):
        """Load model weights and biases from .npy files if they exist."""
        try:
            self.W = np.load(os.path.join(directory, "W.npy"))
            self.U = np.load(os.path.join(directory, "U.npy"))
            self.V = np.load(os.path.join(directory, "V.npy"))
            self.b = np.load(os.path.join(directory, "b.npy"))
            self.c = np.load(os.path.join(directory, "c.npy"))
            print(f"Weights loaded from '{directory}/'")
        except FileNotFoundError:
            print(f"No saved weights found in '{directory}/'. Model initialized with random weights.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example data
    data = np.sin(np.linspace(-np.pi, np.pi, 1000)).reshape(1, -1)

    test_visible = data[:, 500:]
    test_cond = data[:, :500]

    # Initialize the CRBM
    n_visible = 500
    n_cond = 500
    n_hidden = 1000

    crbm = CRBM(n_visible, n_hidden, n_cond)

    # Train the CRBM
    n_epochs = 10
    errors = crbm.train(test_visible, test_cond, n_epochs=n_epochs, batch_size=32, lr=0.001)

    # Visualize reconstruction error
    plt.plot(errors)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Error")
    plt.grid(True)
    plt.show()

    # Test reconstruction
    v_reconstructed = crbm.reconstruct(test_visible, test_cond)

    plt.plot(test_visible[0, :], label="Original (First Feature)", c='r', alpha=0.7)
    plt.plot(v_reconstructed[0, :], label="Reconstructed (First Feature)", c='b', alpha=0.7)
    plt.legend()
    plt.grid(True)
    plt.show()