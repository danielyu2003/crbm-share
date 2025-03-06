import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def extract(df, markerId):
    x_rot = markerId
    y_rot = markerId+'.1'
    z_rot = markerId+'.2'
    w_rot = markerId+'.3'
    return df[[x_rot, y_rot, z_rot, w_rot]][2:].reset_index(drop=True).to_numpy(dtype=float)

def normalize(q):
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q

def lowPassFilter(q):
    x = np.arange(-7 // 2 + 1, 7 // 2 + 1)
    kernel = np.exp(-x**2 / 2)
    kernel /= kernel.sum()
    result = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode='same'), axis=0, arr=q)
    return result

if __name__ == "__main__":
    df = pd.read_csv("Apple_016.csv", skiprows=4, low_memory=False)
    LFOOT = '771BCF0A528711EEC6A1D77FF14871BB'
    quaternions = extract(df, LFOOT)

    # Function to rotate points using a quaternion
    def rotate_with_quaternion(quaternion, points):
        x, y, z, w = quaternion
        rot_matrix = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ])
        return points @ rot_matrix.T

    # Generate the initial sphere
    def generate_sphere(radius=1, resolution=30):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z

    # Function to update the animation
    def update(frame):
        quaternion = quaternions[frame]
        rotated_points = rotate_with_quaternion(quaternion, sphere_points)
        rotated_axes = rotate_with_quaternion(quaternion, axes_lines)

        # Reshape the sphere points
        rx = rotated_points[:, 0].reshape(sphere_x.shape)
        ry = rotated_points[:, 1].reshape(sphere_y.shape)
        rz = rotated_points[:, 2].reshape(sphere_z.shape)
        
        # Clear the old plot
        ax.clear()
        
        # Plot the rotated sphere
        ax.plot_surface(rx, ry, rz, color='b', alpha=0.1)
        
        # Plot the rotation axes
        ax.plot([0, rotated_axes[1, 0]], [0, rotated_axes[1, 1]], [0, rotated_axes[1, 2]], color='r', label='X-axis')
        ax.plot([0, rotated_axes[2, 0]], [0, rotated_axes[2, 1]], [0, rotated_axes[2, 2]], color='g', label='Y-axis')
        ax.plot([0, rotated_axes[3, 0]], [0, rotated_axes[3, 1]], [0, rotated_axes[3, 2]], color='b', label='Z-axis')
        
        ax.text2D(0.05, 0.95, f"Time Step: {frame+1}/{len(quaternions)}", transform=ax.transAxes)

        # Set axis limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.legend()

    print(quaternions.shape)

    input()

    # Normalize the quaternions
    quaternions = np.apply_along_axis(normalize, axis=1, arr=quaternions)
    plt.plot(quaternions[:, 0], label='x')
    plt.plot(quaternions[:, 1], label='y')
    plt.plot(quaternions[:, 2], label='z')
    plt.plot(quaternions[:, 3], label='w')
    plt.legend()

    # Generate the sphere points
    sphere_x, sphere_y, sphere_z = generate_sphere()

    # Flatten sphere points for easier rotation
    sphere_points = np.vstack([sphere_x.ravel(), sphere_y.ravel(), sphere_z.ravel()]).T

    # Define the initial axes
    axes_lines = np.array([
        [0, 0, 0],  # Origin
        [1, 0, 0],  # X-axis
        [0, 1, 0],  # Y-axis
        [0, 0, 1],  # Z-axis
    ])
    # Set up the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    downsample_rate = 10
    quaternions = quaternions[::downsample_rate]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(quaternions), interval=1)

    # Show the animation
    plt.show()