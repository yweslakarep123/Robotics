import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Physical constants
g = 9.8  # gravity
L = 1.5  # length of pendulum
m = 0.5  # mass of pendulum
M = 1.5  # mass of cart

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

# Optimal control parameters
Kp_th = -309.1537837
Kp_x = 1
Kd_th = -30.97613642
Kd_x = 5.80054346

# Initial state vector
state = np.array([th, th_dot, x, x_dot])

# Wind force
wind_force = 1.0  # Constant wind force acting on the pendulum

# Steep mountain terrain function
def terrain(x):
    return 1.0 * np.sin(0.1 * x) + 0.2 * np.sin(0.5 * x)  # Increased amplitude for a steeper slope

# Derivative function
def derivatives(state, t):
    ds = np.zeros_like(state)
    th, th_dot, x, x_dot = state

    # Control input
    F = Kp_th * th + Kd_th * th_dot + Kp_x * (x - x0) + Kd_x * x_dot

    # Disturbances
    if 10 < t < 13:
        F += 5

    # Terrain slope
    terrain_slope = 0.5 * 0.1 * np.cos(0.1 * x) + 0.1 * 0.5 * np.cos(0.5 * x)

    # Equations of motion with wind force
    ds[0] = th_dot
    ds[1] = (F * np.cos(th) - m * L * np.sin(th) * th_dot ** 2 + (M + m) * g * np.sin(th) + wind_force) / ((M + m) * L - m * L * np.cos(th))
    ds[2] = x_dot
    ds[3] = (F - m * L * np.sin(th) * th_dot ** 2 + m * g * np.cos(th) * np.sin(th) + wind_force * np.cos(th) - terrain_slope * g * (M + m)) / ((M + m) - m * np.cos(th) ** 2)

    return ds

# Integrate ODE
solution = integrate.odeint(derivatives, state, t)
ths = solution[:, 0]  # theta output
xs = solution[:, 2]  # x output

# Plot theta and x
fig1 = plt.figure(1)
axs = fig1.subplots(2)
fig1.subplots_adjust(hspace=0.5)
axs[0].set_title("Theta")
axs[1].set_title("x")
for i in range(2):
    axs[i].set_xlim([0, 100])
    axs[i].grid()

axs[0].plot(t, ths)
axs[1].plot(t, xs)

# Animation
fig2 = plt.figure(2, figsize=(12, 6))  # Increase width and height as desired
pxs = L * np.sin(ths) + xs
pys = L * np.cos(ths) + terrain(xs)

ax = fig2.add_subplot(111, autoscale_on=False, xlim=(-10, 10), ylim=(-3, 3))
ax.set_aspect('equal')
ax.grid()

# Draw terrain
terrain_x = np.linspace(-10, 10, 300)
terrain_y = terrain(terrain_x)
ax.plot(terrain_x, terrain_y, 'b-', linewidth=2)

patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))
line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# Increase the size of the cart
cart_width = 0.6  # increased from 0.3 to 0.6
cart_height = 0.4  # increased from 0.2 to 0.4

def init():
    line.set_data([], [])
    time_text.set_text('')
    patch.set_xy((-cart_width / 2, terrain(0) - cart_height / 2))
    patch.set_width(cart_width)
    patch.set_height(cart_height)
    return line, time_text, patch

def animate(i):
    thisx = [xs[i], pxs[i]]
    thisy = [terrain(xs[i]), pys[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    patch.set_x(xs[i] - cart_width / 2)
    patch.set_y(terrain(xs[i]) - cart_height / 2)
    return line, time_text, patch

ani = animation.FuncAnimation(fig2, animate, np.arange(1, len(solution)),
                              interval=25, blit=True, init_func=init)

plt.show()
