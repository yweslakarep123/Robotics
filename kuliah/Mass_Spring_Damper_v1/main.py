import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

# Define the system parameters
m = 1.0  # mass (kg)
b = 0.5  # damping coefficient (N.s/m)
k = 2.0  # spring constant (N/m)

# Define the state-space model
A = np.array([[0, 1], [-k/m, -b/m]])  # System matrix
B = np.array([[0], [1/m]])  # Input matrix
C = np.array([[1, 0]])  # Output matrix (measuring position)
D = np.array([[0]])  # No direct feedthrough

# LQR design parameters
Q = np.array([[10, 0], [0, 1]])  # State cost matrix
R = np.array([[1]])  # Control cost matrix

# Solve for the optimal feedback gain K using the Algebraic Riccati Equation (ARE)
P = linalg.solve_continuous_are(A, B, Q, R)
K = np.dot(np.linalg.inv(R), np.dot(B.T, P))

# Define the closed-loop system dynamics with LQR feedback
def closed_loop_system(x, t):
    # Convert x to a column vector for matrix operations
    x_col = np.array(x).reshape(2, 1)
    # Calculate control input
    u = -np.dot(K, x_col)
    # Calculate derivative
    dxdt = np.dot(A, x_col) + B * u[0,0]  # Extract scalar from u
    # Return as a flattened array of the correct size
    return dxdt.flatten()

# Initial conditions (initial displacement and velocity)
x0 = [1.0, 0.0]  # Start with position=1 and velocity=0
t = np.linspace(0, 10, 500)  # Time vector for simulation

# Solve the ODE system using odeint
sol = odeint(closed_loop_system, x0, t)

# Plot the position and velocity over time
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, sol[:, 0], 'b-', label='Position')
plt.grid()
plt.legend()
plt.ylabel('Position (m)')

plt.subplot(2, 1, 2)
plt.plot(t, sol[:, 1], 'r-', label='Velocity')
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.tight_layout()
plt.savefig('position_velocity_plots.png')
plt.show()

# Prepare the animation for visualization
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 0.5)
ax.grid()
ax.set_title('Mass-Spring-Damper System with LQR Control')
ax.set_xlabel('Position (m)')

# Draw a fixed point for the spring anchor
ax.plot(0, 0, 'ks', markersize=10)

# Draw the spring and mass
spring_line, = ax.plot([], [], 'k-', lw=1)
mass_point, = ax.plot([], [], 'bo', markersize=15)

# Fixed y-coordinate for the system
y_pos = 0

# Function to initialize the animation
def init():
    spring_line.set_data([], [])
    mass_point.set_data([], [])
    return spring_line, mass_point

# Function to update the animation for each frame
def update(frame):
    x_pos = sol[frame, 0]
    # Create spring points (simplified as a line here)
    spring_x = np.linspace(0, x_pos, 20)
    spring_y = np.zeros_like(spring_x)
    
    spring_line.set_data(spring_x, spring_y)
    mass_point.set_data([x_pos], [y_pos])
    return spring_line, mass_point

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=20)

# Save as GIF instead of MP4 since ffmpeg isn't available
from matplotlib.animation import PillowWriter
writer = PillowWriter(fps=30)
ani.save("mass_spring_damper_LQR_simulation.gif", writer=writer)

plt.show()

# Create a state-space plot (phase portrait)
plt.figure()
plt.plot(sol[:, 0], sol[:, 1], 'g-', label='Phase Portrait')
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.title('State-Space Plot (Phase Portrait)')
plt.grid()
plt.legend()
plt.savefig('phase_portrait.png')
plt.show()