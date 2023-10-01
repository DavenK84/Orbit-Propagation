import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

##set up 3d plot
fig, ax = plt.subplots()
ax = plt.axes(projection = "3d")


#label axes
plt.xlabel("x")
plt.ylabel("y")

##scale the axes
plt.axis('scaled')

##Let's plot the earth in a 3D graph
earth_x = 0
earth_y = 0
earth_z = 0
ax.scatter(earth_x,earth_y,earth_z, s = 500)


##let's add a circle to represent a satellite that orbits the earth
##generate theta values from 0 to 2pi (0-360)
theta = np.linspace(0,2*np.pi, 100)

##calculate x, y, z values (z will not change)
#x = earth_x + 10 * np.cos(theta)
#y = earth_y + 10 * np.sin(theta)
#z = earth_z 


##but satellites don't have perfectly circular orbits, they have elliptical orbits
##let's plot a basic ellipse
##x = earth_x + 4 * np.cos(theta)
##y = earth_y + 2 * np.sin(theta)
##z = earth_z

##This works, but satellites have various orbits.
##We're going to model a satellite with a molniya orbit
##This orbit was discovered by Soviet scientists as an alternative to geostationary orbits
##It is highly elliptical
##Which makes it a good candidate to test the kalman filter on
##A typical molniya orbit has the following properties:
##    eccentricity of 0.74
##    Inclination of 63.4 degrees
##    semi major axis of 26,600km (making the semi minor axis 17891km)
##    argument of perigee of 270 degrees 
##    time period of 718 minutes
##    perigee of 600km (point of orbit with lowest altitude)

##Keplers first law states that "The orbit of a planet is an ellipse with the sun at one of the two foci."
##This is true for any object that orbits another under the influence of gravity
##Which means one foci is the centre of the earth, which is 6957km away from the perigee (perigee  + earths radius at pole)
##from this, we can work out the centre of the ellipse, which is (19643,0) for now
##let's redraw our ellipse knowing this information

##draw elipse in only 2 dimensions, and then rotate into 3rd dimension
x = 19643 + 26600 * np.cos(theta) 
y = 0 + 17891 * np.sin(theta) 
z = np.zeros(100)

##stack coords into matrix to prepare for rotation
coords = np.vstack([x,y,z])

##define rotation matrix to create the 63.4 degree angle of inclination
rotation_matrix = np.array([[0.4477591,  0.0000000,  -0.8941543],
                            [0.0000000,  1.0000000,  0.0000000],
                            [0.8941543,  0.0000000,  0.4477591]])

##rotate coords
coords_rotated = np.dot(rotation_matrix, coords)

##unstack and plot
ax.plot(*coords_rotated)

##now we have a simplified molniya orbit plotted.
##Let's study orbital mechanics
##Looking at newtons law of gravitation, we obtain a 2nd order differential equation for the acceleration of the satellite
##it's a long day trying to analytically solve these equations, so I will solve them numerically using the 4th order runge kutta method

##find derivative of current state
def two_body_ode(t, state): 
    GM = 398600 ##gravitational parameter in  km3.s-2 
    r = state[:3] ##r is position vector (first 3 elements in state array)
    a = -GM*r/(np.linalg.norm(r) ** 3) ##calculate acceleration vector (np.linalg.norm returns the length of a vector)
    return np.array([state[3], state[4], state[5], a[0], a[1], a[2]]) ##return derivative state

##4th order runge-kutta method
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h * k1/2)
    k3 = f(t + h/2, y + h * k2/2)
    k4 = f(t + h, y + h * k3)
    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

# Initial conditions for a Molniya orbit
x0, y0, z0 = 46243*np.cos(63.4*np.pi/180), 0, 46243*np.sin(63.4*np.pi/180) ##position of orbit apogee
vx0, vy0, vz0 = 0, 1.501, 0 ##vy0 is found by v = root(GM(2/46243 - 1/26600)), where GM is the gravitational parameter
initial_state = np.array([x0, y0, z0, vx0, vy0, vz0]) ##initial position and velocity of satellite

# Number of steps
steps = 43079 ##each step is a second, period is 718 minutes, so 43080 steps

# Initialize array to store the state at each time step
states = np.zeros((steps+1, 6))
states[0] = initial_state

# Perform the simulation
for i in range(0, steps):
    states[i + 1] = rk4_step(two_body_ode, i, states[i], 1)

plot_data = states[::359] ##purely to plot the data

##plot animation update function
def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])
    return line

##Fetch positional data
x2 = np.array(plot_data[:, 0])
y2 = np.array(plot_data[:, 1])
z2 = np.array(plot_data[:, 2])

##combine positional data
data = np.vstack([x2,y2,z2])
line, = ax.plot(x2, y2, z2, color = 'red')

##Set axes limits and viewing angles, otherwise i have to zoom out and rotate the plot everytime i run the program
ax.set_xlim3d([-25000, 25000])
ax.set_ylim3d([-25000, 25000])
ax.set_zlim3d([-25000, 25000])

ax.view_init(elev=15., azim=-130.)

##animate the plot of the satellite's path
ani = animation.FuncAnimation(fig, update, frames = 43080, fargs=(data, line), interval=10, blit=False, repeat = False)



plt.show()
