import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np
import time

# Number of robots for each line
N_line = 3

# Total number of robots (3 lines + 2 vertical lines)
N = 11 # 15 robots in total

# Instantiate the Robotarium object
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

# Define window boundaries
xybound = np.array([-1.2, 1.2, -0.6, 1.4])

# Function to create horizontal lines
def create_horizontal_lines(N_line, xybound):
    y_values = np.linspace(xybound[2], xybound[3], 4)
    line_top = np.vstack([np.linspace(xybound[0], xybound[1], N_line), np.ones(N_line) * y_values[0]]).T
    line_middle = np.vstack([np.linspace(xybound[0], xybound[1], N_line), np.ones(N_line) * y_values[1]]).T
    line_bottom = np.vstack([np.linspace(xybound[0], xybound[1], N_line), np.ones(N_line) * y_values[2]]).T
    return np.vstack([line_top, line_middle, line_bottom])

def create_vertical_line(N_line, xybound, x_position, y_start, y_end):
    """
    Create a vertical line at a specified x position between two y coordinates.

    Parameters:
    N_line (int): Number of robots for the line.
    xybound (list): List of 4 floats defining the window boundaries [xmin, xmax, ymin, ymax].
    x_position (float): The x-coordinate where the vertical line should be placed.
    y_start (float): The starting y-coordinate of the line.
    y_end (float): The ending y-coordinate of the line.

    Returns:
    numpy.ndarray: An array of target positions for the vertical line.
    """
    y_values = np.linspace(y_start, y_end, N_line)
    return np.vstack([np.ones(N_line) * x_position, y_values]).T


y_start_top = 0.4
y_end_middle = 0
y_start_middle = -0.3
y_end_bottom = -0.6

x_goal_horizontal = create_horizontal_lines(N_line, xybound).T
# Create target positions for vertical lines
x_goal_left_vertical = create_vertical_line(1, xybound, x_position=-1.2, y_start=y_start_top, y_end=y_end_middle).T
x_goal_right_vertical = create_vertical_line(1, xybound, x_position=1.2, y_start=y_start_middle, y_end=y_end_bottom).T

# Combine target positions for horizontal and vertical lines
x_goal = np.hstack([x_goal_horizontal, x_goal_left_vertical, x_goal_right_vertical])


# Ensure x_goal matches the number of robots
x_goal = x_goal[:, :N]  # Adjust the shape of x_goal to match the number of robots



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
    
    # Gain for the speed controller
    K = 50

    # Position controller
    dxi = K * si_position_controller(x_si, x_goal)

    # Apply barrier certificates
    dxi = si_barrier_cert(dxi, x_si)

    # Map to unicycle dynamics
    dxu = si_to_uni_dyn(dxi, x)

    # Set velocities and iterate the simulation
    r.set_velocities(np.arange(N), dxu)
    r.step()


# Update the caption for the final formation
finished_caption = "Robots formed the letter 'S'"
finished_label = r.axes.text(0, 0, finished_caption, fontsize=font_height_points, color='k', fontweight='bold', horizontalalignment='center', verticalalignment='center', zorder=20)
r.step()
time.sleep(10)

# End the simulation properly
r.call_at_scripts_end()
