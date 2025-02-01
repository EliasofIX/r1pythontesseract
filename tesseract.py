import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

# Generate tesseract vertices and edges
vertices = list(itertools.product([-1, 1], repeat=4))
edges = []
for vertex in vertices:
    for dim in range(4):
        neighbor = list(vertex)
        neighbor[dim] *= -1
        neighbor = tuple(neighbor)
        if neighbor in vertices:
            edge = (vertex, neighbor)
            if edge not in edges and (neighbor, vertex) not in edges:
                edges.append(edge)

# Projection parameters
d = 2  # Viewpoint distance along the w-axis

def project_4d_to_3d(point4d):
    x, y, z, w = point4d
    factor = 1 / (d - w)
    return (x * factor, y * factor, z * factor)

# Set up figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('4D Ball Bouncing in a Tesseract')

# Plot tesseract edges
for edge in edges:
    start = project_4d_to_3d(edge[0])
    end = project_4d_to_3d(edge[1])
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color='gray', alpha=0.3)

# Initialize ball position and velocity in 4D
pos = np.array([0.0, 0.0, 0.0, 0.0])
velocity = np.array([0.1, 0.13, 0.15, 0.12])  # Initial velocity

# Create scatter plot for ball
ball = ax.scatter([], [], [], color='darkred', s=50, edgecolors='black')

def update(frame):
    global pos, velocity
    
    # Update position
    dt = 0.05
    pos += velocity * dt
    
    # Bounce off hypercube walls
    for i in range(4):
        if pos[i] >= 1 or pos[i] <= -1:
            velocity[i] *= -1
            pos[i] = np.clip(pos[i], -1, 1)
    
    # Project to 3D and update visualization
    projected = project_4d_to_3d(pos)
    factor = 1 / (d - pos[3])
    
    # Update ball position and size
    ball._offsets3d = ([projected[0]], [projected[1]], [projected[2]])
    ball.set_sizes([50 * factor])
    
    return ball,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=40, blit=True)
plt.show()
