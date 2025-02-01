# r1pythontesseract
Tesseract with bouncing ball inside fully programmed by Deepseek R1
Remarkable stuff right there. Here is a video showcasing the product: 

Prompt: write a python script of a ball bouncing inside a tesseract
Heres the reasoning process:


Okay, I need to write a Python script that shows a ball bouncing inside a tesseract. Hmm, a tesseract is a 4D hypercube, but since we can't visualize 4D directly, I have to project it into 3D, which then gets rendered on a 2D screen. So the first step is figuring out how to represent a tesseract in a way that can be displayed.

I remember that a tesseract can be represented by its vertices in 4D space, each with coordinates (x, y, z, w) where each coordinate is either -1 or 1. Then, to project it into 3D, I can use a projection method, probably orthographic projection by ignoring the fourth dimension or using some perspective. But maybe a simple approach is to just drop the w-coordinate, which would give a 3D shadow. However, that might not look dynamic. Alternatively, a perspective projection where the 4th dimension affects the scaling based on distance. But perhaps for simplicity, I'll start by dropping the w-coordinate to get the 3D structure.

Next, the ball's movement. The ball is moving in 4D space, bouncing off the walls of the tesseract. The tesseract's boundaries in each dimension are from -1 to 1, so the ball's position in each of the four dimensions needs to stay within those bounds. When it hits a boundary (position >=1 or <=-1 in any dimension), the velocity in that dimension reverses. That's similar to how a ball bounces in 3D, just extended to 4D.

But how to represent the ball's position in 4D and then project it to 3D for visualization. The projection could be similar to how the tesseract is projected. So the ball's 3D position for display would be its 4D coordinates after projection. If we drop the w-coordinate, then the ball's position in 3D would just be (x, y, z). But maybe to make it more interesting, use a perspective projection where the w-coordinate affects the position. For example, a projection that uses a point as the "camera" in 4D space, projecting onto a 3D hyperplane. That might complicate things, though. Maybe start with a simple orthographic projection by ignoring the w-coordinate, then later adjust if needed.

So the steps are:

1. Create a tesseract's vertices and edges in 4D.

2. Project these into 3D (by dropping w) for visualization.

3. Draw the tesseract edges as lines in 3D space using matplotlib or another library.

4. Simulate a ball moving in 4D space, bouncing off the hypercube's walls.

5. Project the ball's 4D position into 3D and display it as a sphere.

But which library to use? Matplotlib's 3D plotting might be sufficient, though it's not the most performant for animations. Alternatively, using Pygame with 3D projections, but that's more complex. Since the user asked for Python and simplicity, matplotlib with animation is probably the way to go.

So, using matplotlib's mplot3d to plot the tesseract edges and the ball. The ball's position will be updated in each frame, changing based on its velocity in 4D. Each dimension's velocity reverses when the position exceeds ±1.

Now, coding steps:

- Define the 4D vertices of the tesseract. There are 16 vertices, each with (x,y,z,w) where each is ±1.

- Generate edges by connecting vertices that differ in exactly one coordinate.

- Project vertices to 3D by dropping the w-coordinate (or using another projection method).

- Set up the 3D plot in matplotlib.

- Initialize the ball's position and velocity in 4D.

- In each animation frame, update the ball's position, check for boundary collisions, reverse velocity if needed.

- Project the ball's 4D position to 3D (e.g., (x, y, z)) and update the scatter plot marker's position.

Wait, but if we just drop w, then the ball's 3D position is just x, y, z. But the ball is moving in 4D, so even if the 3D projection is static (w is fixed), the ball's w-component is changing. However, in the projection, the ball would appear to move in x, y, z as those components change, but also, if the w-component is changing, but we're ignoring it, then the 3D position is only based on x, y, z. But the ball is moving in 4D, so all four coordinates are changing. For example, if the ball is moving along the w-axis, in the 3D projection (x,y,z) would remain constant, but the ball's w is changing. But since we're not showing w, the ball would appear stationary. That's not ideal. So perhaps the projection needs to include some effect from the w-coordinate.

Alternatively, maybe the projection should be a 4D to 3D perspective projection. For example, similar to projecting 3D to 2D with perspective, but one more dimension. So, a 4D point (x,y,z,w) could be projected into 3D by dividing by some function of the w-coordinate. But how?

Wait, in 3D to 2D perspective projection, we divide x and y by (d - z) where d is the distance from the viewer. For 4D to 3D, maybe a similar approach: project (x,y,z,w) into 3D by dividing by (d - w), resulting in (x/(d - w), y/(d - w), z/(d - w)). This would give a perspective projection where the w-coordinate affects the scaling. The parameter d is the viewer's position along the w-axis. Choosing d=2, for example.

This way, as the ball moves in the w-direction, its projection into 3D would scale accordingly. When the ball is at w=1, the divisor is d - w = 2-1=1, so the projection is (x,y,z). If the ball moves to w=0, the divisor is 2, so the projection is (x/2, y/2, z/2), making the ball appear smaller. At w approaching d (2), the divisor approaches zero, leading to large projections, but since the ball is constrained within -1 to 1 in all dimensions, w can't reach 2. So that's safe.

So using a perspective projection would make the ball's movement in the 4th dimension affect its apparent position and size in 3D. That adds more visual interest and shows the 4D movement.

So, the projection function would take a 4D point (x,y,z,w) and return a 3D point (x/(d - w), y/(d - w), z/(d - w)). Then, d is a parameter we can set, say 3 for example, to have a reasonable scaling.

But then, the tesseract edges would also need to be projected using the same method. So instead of just dropping w, each vertex is projected using this perspective projection. That would make the tesseract's appearance change dynamically if we rotate it in 4D, but in this case, the tesseract is static, and the ball is moving. Wait, no—the tesseract is fixed in 4D space, so projecting its edges with the same projection as the ball would make the ball's position relative to the tesseract's edges accurate in the projection.

Wait, but the tesseract's edges are in 4D. If we project the tesseract's edges using the same perspective projection, then the ball's position in the projected 3D space would correctly appear inside the projected tesseract.

But then, how to draw the tesseract's edges? Each edge is a line between two 4D points. When projected, those two points become 3D points, so the edge is drawn between their projections.

So, the steps now are:

1. Define the tesseract's 4D vertices and edges.

2. Project each vertex using the perspective projection to 3D.

3. Draw the edges between the projected vertices.

4. The ball's 4D position is updated each frame, and its projection is computed to 3D for display.

This approach would make the ball's movement in all four dimensions visible in the 3D projection, as changes in w would affect the scaling.

Now, implementing this in Python with matplotlib.

First, generate all the vertices of the tesseract. Each coordinate is either -1 or 1, so there are 16 vertices. Then, edges connect vertices that differ in exactly one coordinate. So for each vertex, we can compare with all others and connect if their Hamming distance is 1. Alternatively, generate all possible edges by iterating through each vertex and flipping each coordinate to get connected vertices.

But generating the edges can be done by checking all pairs of vertices and seeing if they differ in exactly one coordinate. However, that's O(n²), which is manageable for 16 vertices.

Alternatively, for each vertex (a list of four coordinates), for each axis (0 to 3), flip that axis's coordinate to get the connected vertex. So for vertex (1,1,1,1), flipping the first axis gives (-1,1,1,1), which is connected. This way, each vertex has four edges (one for each axis), but since edges are bidirectional, we can avoid duplicates by generating them in a set.

So code to generate edges:

vertices = list(itertools.product([-1,1], repeat=4))

edges = []

for vertex in vertices:

for dim in range(4):

neighbor = list(vertex)

neighbor[dim] *= -1

neighbor = tuple(neighbor)

if neighbor in vertices:

if (neighbor, vertex) not in edges and (vertex, neighbor) not in edges:

edges.append( (vertex, neighbor) )

But maybe there's a smarter way, but for 16 vertices, it's manageable.

Once edges are generated, project each vertex using the perspective projection. For example, with d=3:

def project_4d_to_3d(point4d, d=3):

x, y, z, w = point4d

factor = 1 / (d - w)

return (x * factor, y * factor, z * factor)

Then, for each edge in 4D, project both endpoints and draw a line between their 3D projections.

Now, for the animation part. The ball's position is a 4D vector (x, y, z, w) with velocity (vx, vy, vz, vw). Each frame, the position is updated by adding velocity multiplied by dt. Then, check if any coordinate exceeds ±1. If so, reverse velocity in that dimension and adjust position to be within bounds (to prevent getting stuck).

But for simplicity, dt can be 0.01 or similar, and each frame steps the position by velocity. Then, when a coordinate reaches 1 or -1, the velocity in that dimension reverses.

Now, the ball's 4D position is projected to 3D using the same projection as the tesseract. So the ball's 3D position is (x/(d - w), y/(d - w), z/(d - w)), and its size could be scaled based on the projection factor. For example, a larger w (closer to d) would make the ball appear larger, but since the ball is within -1 <= w <=1, and d is 3, d - w is between 2 and 4, so factor is between 0.25 and 0.5. Wait, factor is 1/(d - w), so when w is 1, factor is 1/(3-1)=0.5, when w is -1, factor is 1/(3 - (-1))=0.25. So the ball's size would scale accordingly. So maybe the size in the 3D plot can be adjusted based on the projection factor to simulate perspective.

Alternatively, the size could remain constant, but that might not look right. Hmm, but in matplotlib's 3D scatter plot, the size is in points, which doesn't account for 3D perspective. So maybe it's better to not adjust the size, or perhaps scale the marker size based on the factor. For example, multiply the base size by the factor.

But for simplicity, perhaps just keep the ball's size constant. Or adjust it for effect.

Putting this all together:

- Set up the 3D axes in matplotlib.

- Plot all the edges of the tesseract as lines in 3D after projection.

- Initialize the ball's position and velocity in 4D.

- In the animation function, update the ball's position, check for collisions, project to 3D, and update the ball's marker position.

Potential issues:

- The tesseract's projected edges might look static, but in reality, if we are not rotating them in 4D, they remain the same. However, the ball moving in 4D would appear to move through the projected tesseract.

Another consideration: when the ball moves in the w dimension, its projection's x, y, z coordinates change even if x, y, z in 4D are fixed. For example, if the ball moves from (0,0,0,0) to (0,0,0,1), the projection would go from (0/(3-0), 0/(3-0), 0/(3-0)) = (0,0,0) to (0/(3-1), 0/(3-1), 0/(3-1)) = (0,0,0). Wait, that's the same. Hmm, maybe the example is bad. Suppose the ball is at (1,0,0,0). Projected as (1/3, 0, 0). If it moves to (1,0,0,1), projected as (1/(3-1), 0, 0) = 0.5, so the x-coordinate increases. So movement in the w dimension affects the scaling of the other coordinates. So the ball moving in w will appear to move in x, y, z even if their 4D coordinates are fixed. So that's good; the movement in w is visible.

So, implementing the projection correctly is essential to show the 4D movement.

Now, code outline:

Import necessary modules: matplotlib.pyplot, matplotlib.animation, mpl_toolkits.mplot3d, itertools.

Generate vertices and edges.

Set up the figure and 3D axis.

Plot the tesseract edges.

Initialize the ball's position and velocity in 4D.

Create a scatter plot for the ball.

Define the animation function that updates the ball's position and projects it.

Implement collision detection in 4D.

Test the code.

Potential adjustments:

- The velocity initialization. Maybe start with some velocity in all four dimensions.

- The projection parameter d. Tweak d to get a better visual effect.

- The speed of the animation; adjust the interval and dt.

- Edge colors and styles for better visibility.

- The ball's color and size.

Now, possible code:

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

d = 3  # Viewpoint distance along the w-axis

def project_4d_to_3d(point4d):

x, y, z, w = point4d

factor = 1 / (d - w)

return (x * factor, y * factor, z * factor)

# Set up the figure and 3D axis

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-1, 1)

ax.set_ylim(-1, 1)

ax.set_zlim(-1, 1)

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_zlabel('Z')

# Plot the tesseract edges

for edge in edges:

start = project_4d_to_3d(edge[0])

end = project_4d_to_3d(edge[1])

line = ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='gray', alpha=0.5)

# Initialize the ball's position and velocity in 4D

pos = np.array([0.0, 0.0, 0.0, 0.0])

velocity = np.array([0.1, 0.2, 0.15, 0.12])  # Velocity in all four dimensions

# Create a scatter plot for the ball

ball = ax.scatter([], [], [], color='red', s=50)

def update(frame):

global pos, velocity

# Update position

dt = 0.05

pos += velocity * dt

# Bounce off the walls in each dimension

for i in range(4):

if pos[i] >= 1 or pos[i] <= -1:

velocity[i] *= -1

# Adjust position to prevent getting stuck

pos[i] = np.clip(pos[i], -1, 1)

# Project the ball's 4D position to 3D

projected = project_4d_to_3d(pos)

ball._offsets3d = (np.array([projected[0]]), np.array([projected[1]]), np.array([projected[2]]))

return ball,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)

plt.show()

Wait, but in the projection function, if w approaches d=3, the factor becomes large. But since the ball's position in w is constrained between -1 and 1, d - w ranges from 4 (when w=-1) to 2 (when w=1). So factor ranges from 0.25 to 0.5. So the projected coordinates are scaled by 0.25 to 0.5. So the ball's position in 3D would have coordinates up to 0.5*1=0.5 (since in 4D, each axis is up to 1). But in the current code, the axes are set to -1 to 1. That's a problem because the projected coordinates will be between -0.5 and 0.5. So the axes limits should be adjusted to -0.5 to 0.5. Or change d to a smaller value so that the projection fills the axes.

Alternatively, set the axes limits based on the maximum possible projected values. The maximum projected coordinate would be when a 4D coordinate is 1 in x and w is -1, so factor is 1/(3 - (-1))=0.25. So x_proj = 1 * 0.25 = 0.25. Similarly, if x is 1 and w is 1, factor is 0.5, so x_proj = 0.5. So the maximum projected coordinate is 0.5, minimum is -0.5. So setting the axes limits from -0.5 to 0.5 would make the tesseract edges and ball visible within the plot.

So changing the axes limits:

ax.set_xlim(-0.5, 0.5)

ax.set_ylim(-0.5, 0.5)

ax.set_zlim(-0.5, 0.5)

But wait, the edges of the tesseract when projected. For example, a vertex (1,1,1,1) would project to (1/(3-1), 1/(3-1), 1/(3-1)) = (0.5, 0.5, 0.5). Similarly, (-1,-1,-1,-1) would project to (-1/(3 - (-1)), ...) = (-0.25, -0.25, -0.25). So the maximum projected coordinate is 0.5 and minimum -0.25. But other vertices may have different combinations. For example, (1,1,1,-1) would project to (1/(3 - (-1)), etc.) = (0.25, 0.25, 0.25). So the overall limits should probably be from -0.5 to 0.5, which would accommodate all possible projections.

But in the code above, when we project the edges, their coordinates are scaled down. So the original code's axes limits from -1 to 1 would make the plotted edges very small. So adjusting the axes limits is crucial.

Alternatively, we can set d=2. Then, when w=1, factor=1/(2-1)=1, so the projected coordinates can be up to 1 (if x=1). When w=-1, factor=1/(2 - (-1))=1/3 ≈ 0.333. So the projected coordinates vary between -1 and 1. So setting d=2 and axes limits from -1 to 1 would work better.

Let me adjust the projection parameter d to 2.

So:

d = 2

Then, the projection factor for w=1 is 1/(2-1)=1, so x=1 in 4D would project to x=1 in 3D. For w=-1, factor=1/(2 - (-1))=1/3. So coordinates in 3D would range from -1 to 1 in x, y, z. So the axes limits can be set to -1.5 to 1.5 to have some padding, or -1 to 1.

But let's see. If a 4D point is (1,0,0,1), then projection is (1/(2-1), 0, 0) = (1, 0, 0). If another point is (1,0,0,-1), projection is (1/(2 - (-1)), 0, 0) ≈ (0.333, 0, 0). So the axes limits from -1.5 to 1.5 would show everything. Or maybe -1.2 to 1.2 to have some margin.

So changing d to 2 and adjusting axes:

ax.set_xlim(-1.2, 1.2)

ax.set_ylim(-1.2, 1.2)

ax.set_zlim(-1.2, 1.2)

This way, the edges at maximum projection (up to 1) fit within the axes with padding.

Also, in the ball's projection, if it's moving in w, the scaling would vary between 1 (when w=1) and 0.333 (when w=-1). So the ball would appear larger when moving in the +w direction and smaller in the -w direction. To make this visible, perhaps adjusting the marker size based on the projection factor. For example, s=50 * factor.

But in matplotlib's 3D scatter plot, the 's' parameter is the marker size in points^2, and it doesn't automatically scale with the 3D view. So increasing the size when the ball is further away (but in this case, when w is higher, the ball is projected closer to the viewer, so larger). Wait, the projection factor is 1/(d - w). For d=2, when w=1, factor is 1, so the marker size is 50 * 1 =50. When w=-1, factor=1/3, so size 50/3 ≈16.666. So the ball's size would decrease as it moves towards w=-1, giving a sense of depth.

So in the update function:

projected = project_4d_to_3d(pos)

factor = 1 / (d - pos[3])

ball.set_sizes([50 * factor])

But in matplotlib's 3D scatter plot, the size is not dynamically updated based on distance, so changing the size might not give a perfect perspective effect, but it can help indicate depth.

So modifying the code in the update function:

ball.set_sizes([50 * factor])

But wait, the 'set_sizes' method of the scatter plot expects an array-like of sizes. So passing [50 * factor] should work.

Putting this all together, here's the revised code:

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

# Set up the figure and 3D axis

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-1.2, 1.2)

ax.set_ylim(-1.2, 1.2)

ax.set_zlim(-1.2, 1.2)

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_zlabel('Z')

# Plot the tesseract edges

for edge in edges:

start = project_4d_to_3d(edge[0])

end = project_4d_to_3d(edge[1])

ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='gray', alpha=0.5)

# Initialize the ball's position and velocity in 4D

pos = np.array([0.0, 0.0, 0.0, 0.0])

velocity = np.array([0.1, 0.2, 0.15, 0.12])  # Velocity in all four dimensions

# Create a scatter plot for the ball

ball = ax.scatter([], [], [], color='red', s=50)

def update(frame):

global pos, velocity

# Update position

dt = 0.05

pos += velocity * dt

# Bounce off the walls in each dimension

for i in range(4):

if pos[i] >= 1 or pos[i] <= -1:

velocity[i] *= -1

# Adjust position to prevent getting stuck

pos[i] = np.clip(pos[i], -1, 1)

# Project the ball's 4D position to 3D

projected = project_4d_to_3d(pos)

factor = 1 / (d - pos[3])

ball._offsets3d = (np.array([projected[0]]), np.array([projected[1]]), np.array([projected[2]]))

ball.set_sizes([50 * factor])

return ball,

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)

plt.show()

Testing this code should show a red ball bouncing inside a projected tesseract. The ball's size changes as it moves along the w-axis, giving a sense of depth. The tesseract edges are static since we're not rotating them, but the ball's movement in all four dimensions is visualized through its projected position and size.

Potential issues to watch for:

- The projection might be incorrect if the formula is wrong. Need to make sure the projection correctly calculates x/(d - w), etc.

- The velocity directions and collision detection must correctly handle all four dimensions.

- Performance issues with matplotlib's animation, but for a simple animation, it should work.

This should create a basic visualization of a 4D ball bouncing inside a tesseract, projected into 3D.
