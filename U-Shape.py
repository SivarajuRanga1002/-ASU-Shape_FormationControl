import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Number of robots for each component of the "A"
N = 10 

# # Update the total number of robots
# N = 2 * N_line + N_horizontal + N_top

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

def rotate_points(points, angle):
    """
    Rotate a set of points by a given angle (in radians).

    Parameters:
    points (numpy.ndarray): An array of points to be rotated.
    angle (float): The angle in radians by which to rotate the points.

    Returns:
    numpy.ndarray: The rotated points.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return np.dot(points - np.mean(points, axis=0), rotation_matrix) + np.mean(points, axis=0)

def create_C_formation(N_curve, xybound_c):
    """
    Create positions for the letter 'C' formation.

    Parameters:
    N_curve (int): Number of robots for the curve of the 'C'.
    xybound (list): List of 4 floats defining the window boundaries [xmin, xmax, ymin, ymax].

    Returns:
    numpy.ndarray: An array of target positions for the 'C' formation.
    """

    # Define the curve of the 'C'
    theta = np.linspace(0, np.pi, N_curve)
    curve_x = xybound[0] + (xybound[1] - xybound[0]) * 0.5 + (xybound[1] - xybound[0]) * 0.4 * np.cos(theta)
    curve_y = xybound[2] + (xybound[3] - xybound[2]) * 0 + (xybound[3] - xybound[2]) * 0.8 * np.sin(theta)
    curve = np.vstack([curve_x, curve_y]).T
    curve = rotate_points(curve, np.pi)  # Rotate by 180 degrees


    return curve

# Target positions for "A" formation
x_goal = create_C_formation(N, xybound).T
# C_shape = create_C_formation(N_curve, xybound_c)

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
finished_caption = "Robots formed the letter 'U'"
finished_label = r.axes.text(0, 0, finished_caption, fontsize=font_height_points, color='k', fontweight='bold', horizontalalignment='center', verticalalignment='center', zorder=20)
r.step()
time.sleep(10)

# End the simulation properly
r.call_at_scripts_end()
