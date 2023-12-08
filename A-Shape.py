import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Number of robots for each component of the "A"
N_line = 4  # For the slanted lines
N_horizontal = 1  # For the horizontal line
N_top = 1  # For the top point

# Update the total number of robots
N = 2 * N_line + N_horizontal + N_top

# Instantiate the Robotarium object with the updated number of robots
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# Safety and plotting parameters
safety_radius = 0.15
CM = np.random.rand(N, 3)  # Random colors for the robots
safety_radius_marker_size = determine_marker_size(r, safety_radius)
font_height_meters = 0.1
font_height_points = determine_font_size(r, font_height_meters)

# Initial plots
x = r.get_poses()
g = r.axes.scatter(x[0, :], x[1, :], s=np.pi/4 * safety_radius_marker_size, marker='o', facecolors='none', edgecolors=CM, linewidth=7)
r.step()

# Barrier and controller setup
si_barrier_cert = create_single_integrator_barrier_certificate()
si_position_controller = create_si_position_controller()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Define the two slanted lines for the "A" formation
xybound = np.array([-1.2, 1.2, -1, 1])  # Window boundaries

# First slanted line
start_point_1 = np.array([xybound[0], xybound[2]])
end_point_1 = np.array([0, xybound[3]])
line_1_x = np.linspace(start_point_1[0], end_point_1[0], N_line + 1)[:-1]
line_1_y = np.linspace(start_point_1[1], end_point_1[1], N_line + 1)[:-1]
line_1 = np.vstack([line_1_x, line_1_y]).T

# Second slanted line
start_point_2 = np.array([xybound[1], xybound[2]])
end_point_2 = np.array([0, xybound[3]])
line_2_x = np.linspace(start_point_2[0], end_point_2[0], N_line + 1)[:-1]
line_2_y = np.linspace(start_point_2[1], end_point_2[1], N_line + 1)[:-1]
line_2 = np.vstack([line_2_x, line_2_y]).T

# Top point of the "A"
top_point = np.array([[0, xybound[3]]])

# Calculate the y-coordinate for the horizontal line to be between the top and bottom of the slanted lines
y_horizontal = np.average([line_1_y[1], line_1_y[-1]])
# Define the horizontal line span based on the slanted lines
horizontal_line_span = (line_2_x[0] - line_1_x[0]) / 2  # Half the distance between the two slanted lines


# Calculate the midpoint of the horizontal line
horizontal_line_x_midpoint = (line_1_x[-1] + line_2_x[-1]) / 2
horizontal_line_y_midpoint = y_horizontal

# Create a single point representing the center of the horizontal line
horizontal_line_center = np.array([[horizontal_line_x_midpoint, horizontal_line_y_midpoint]])


#Working code for the letter A (Has 3 robots in the center but not aligned perfectly so doesnt complete the shape A)
# # Adjust the horizontal line span to fit between the slanted lines correctly
# horizontal_line_x_start = (line_1_x[-1] + line_2_x[-1]) / 2 - horizontal_line_span / 2
# horizontal_line_x_end = (line_1_x[-1] + line_2_x[-1]) / 2 + horizontal_line_span / 2
# horizontal_line_x = np.linspace(horizontal_line_x_start, horizontal_line_x_end, N_horizontal)
# horizontal_line_y = np.ones(N_horizontal) * y_horizontal
horizontal_line = np.vstack([horizontal_line_x_midpoint, horizontal_line_y_midpoint]).T

# # Horizontal line of the "A"
# y_horizontal = (line_1_y[1] + top_point[0, 1]) / 2  # Adjust the index to get the second y-coordinate from the bottom of line_1
# horizontal_line_span = 0.5  # Adjust this value as needed
# horizontal_line_x = np.linspace(-horizontal_line_span / 2, horizontal_line_span / 2, N_horizontal) + start_point_1[0] + (end_point_1[0] - start_point_1[0]) / 2
# horizontal_line_y = np.ones(N_horizontal) * y_horizontal
# horizontal_line = np.vstack([horizontal_line_x, horizontal_line_y]).T


# Combine all parts for the "A" formation
A_shape = np.concatenate([line_1, horizontal_line, top_point, line_2], axis=0)

# Target positions for "A" formation
x_goal = A_shape.T

# Simulation loop
while True:
    # Get the current poses of the robots
    x = r.get_poses()
    g.set_offsets(x[:2, :].T)
    g.set_sizes([determine_marker_size(r, safety_radius)])
    
    # Convert to single integrator states
    x_si = uni_to_si_states(x)

    # Calculate the error between current and goal positions
    errors = x_goal - x_si
    if np.linalg.norm(errors) < 0.01 * N:
        break

    
    #Gain for the speed controller
    K = 50

    # Position controller
    dxi = K*si_position_controller(x_si, x_goal)

    # Apply barrier certificates
    dxi = si_barrier_cert(dxi, x_si)

    # Map to unicycle dynamics
    dxu = si_to_uni_dyn(dxi, x)

    # Set velocities and iterate the simulation
    r.set_velocities(np.arange(N), dxu)
    r.step()

# Update the caption for the final formation
finished_caption = "Robots formed the letter 'A'"
finished_label = r.axes.text(0, 0, finished_caption, fontsize=font_height_points, color='k', fontweight='bold', horizontalalignment='center', verticalalignment='center', zorder=20)
r.step()
time.sleep(10)

# End the simulation properly
r.call_at_scripts_end()
