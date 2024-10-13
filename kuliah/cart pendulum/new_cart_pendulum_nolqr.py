import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.linalg import eigvals
from control import ctrb, place
from matplotlib.animation import FuncAnimation

# Given physical constants
g = 9.81  # gravity (m/s^2)
L = 0.36  # length of pendulum rod (m)
m = 0.23  # mass of pendulum (kg)
M = 2.4   # mass of cart (kg)
b = 0.1   # friction coefficient between the cart base

# Simulation parameters
dt = 0.05  # time step
Tmax = 100  # total simulation time
t = np.arange(0.0, Tmax, dt)  # time array

# Initial conditions
th_dot = 0.0  # initial angular velocity
th = np.pi / 10  # initial angle
x = 0.0  # initial cart position
x0 = 0.0  # desired cart position
x_dot = 0.0  # initial cart velocity

# State space representation
# State vector: [theta, theta_dot, x, x_dot]
A = np.array([[0, 1, 0, 0],
              [(m+M)*-g/(M*L), 0, 0, b/(M*L)],
              [0, 0, 0, 1],
              [m*g/M, 0, 0, -b/M]])

B = np.array([[0],
              [-1/(M*L)],
              [0],
              [1/M]])

C = np.eye(4)  # Output the full state
D = np.zeros((4, 1))

# Stability analysis: eigenvalues of A
eigenvalues = eigvals(A)
print("Eigenvalues of the system (A matrix):")
print(eigenvalues)

# Controllability analysis
controllability_matrix = ctrb(A, B)
rank_of_controllability_matrix = np.linalg.matrix_rank(controllability_matrix)
print("\nControllability Matrix:")
print(controllability_matrix)
print("\nRank of the Controllability Matrix:", rank_of_controllability_matrix)
if rank_of_controllability_matrix == A.shape[0]:
    print("The system is controllable.")
else:
    print("The system is not controllable.")

# Manually designed state feedback gains (using pole placement)
desired_poles = [-1, -1.5, -2, -2.5]
K_manual = place(A, B, desired_poles)
print("\nManual Gain Matrix (K):")
print(K_manual)

# Define the system dynamics with manual control input
def derivatives(state, t):
    x1, x2, x3, x4 = state
    state_vec = np.array([[x1], [x2], [x3], [x4]])
    u = -K_manual @ (state_vec - np.array([[0], [0], [x0], [0]]))
    dxdt = (A @ state_vec + B @ u).flatten()
    return dxdt

# Initial state vector
state = np.array([th, th_dot, x, x_dot])

# Integrate ODE
solution = integrate.odeint(derivatives, state, t)
ths = solution[:, 0]  # theta output
ths_dot = solution[:, 1]
xs = solution[:, 2]  # x output
xs_dot = solution[:, 3]

# Plot the response for each state
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.5)
state_labels = ['Theta (rad)', 'Theta_dot (rad/s)', 'X (m)', 'X_dot (m/s)']
state_data = [ths, ths_dot, xs, xs_dot]

for i, ax in enumerate(axs):
    ax.plot(t, state_data[i])
    ax.set_title(state_labels[i])
    ax.set_xlim([0, Tmax])
    ax.grid()

plt.tight_layout()
plt.show()

# Animation
fig2 = plt.figure(2, figsize=(12, 6))
pxs = L * np.sin(ths) + xs
pys = L * np.cos(ths)

ax = fig2.add_subplot(111, autoscale_on=False, xlim=(-10, 10), ylim=(-3, 3))
ax.set_aspect('equal')
ax.grid()

# Increase the size of the cart
cart_width = 0.6  # increased from 0.3 to 0.6
cart_height = 0.4  # increased from 0.2 to 0.4

patch = plt.Rectangle((0, 0), cart_width, cart_height, color='blue')
ax.add_patch(patch)
line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    patch.set_xy((-cart_width / 2, -cart_height / 2))
    return line, time_text, patch

def animate(i):
    thisx = [xs[i], pxs[i]]
    thisy = [0, pys[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    patch.set_x(xs[i] - cart_width / 2)
    patch.set_y(-cart_height / 2)
    return line, time_text, patch

ani = FuncAnimation(fig2, animate, np.arange(1, len(solution)),
                              interval=25, blit=True, init_func=init)

plt.show()
