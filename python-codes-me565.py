import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# actual square wave
x_sq = [-2, -1, -1, 1, 1, 2]
y_sq = [0, 0, 1, 1, 0, 0]
def fourier_sq(x, n):
    '''fourier series approx to square wave with n terms'''
    # first term
    y = 0.5*np.ones_like(x)
    sign = -1
    # other terms m = 1, 3, 5, 7
    for m in range(1, 2*n, 2):
        # alternate +ve and -ve sign
        sign = -sign
        y += sign*2/m/np.pi * np.cos (m*np.pi*x / 2)
    return y
xSM = np.arange(-2, 2, 0.01)
xL = np.arange(-2, 2, 0.2)
ySM = fourier_sq(xSM, 10)
yL = fourier_sq(xL, 10)
fig = plt.figure()
plt.plot(x_sq, y_sq, 'b', label='square wave')
plt.plot(xSM, ySM, 'c', label='10-term fourier series\nstep=0.01 (small)')
plt.plot(xL, yL, 'xm', label='10-term fourier series\nstep=0.2 (large)')
plt.xlabel('time, t'), plt.ylabel('signal, y')
plt.title('Fourier Series of Square Wave')
plt.legend(loc='center', bbox_to_anchor=[1,.8,.5,0])
plt.grid()
plt.show()

######################
######################

import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

# laplace equation
# ∂U/∂t = 𝛼²∇²U
# on a circular disk with 
# R=1, 𝛼=1
# domain -1.5 to 1.5 in x and y
points = np.linspace(-1.5, 1.5, 300)
y, x = np.meshgrid(points, points)
dx, dy = (points[1] - points[0], points[1] - points[0])
def outsidecircle(xarr, yarr):
    '''Operates on x,y coordinate arrays. Returns boolean array whose elements are True if x,y lie outside unit circle'''
    return xarr**2 + yarr**2 > 1
def lefttest(xarr, yarr):
    '''Boolean array whose elements are True if x,y on left boundary of unit circle'''
    return np.logical_and(xarr < 0, outsidecircle(xarr, yarr))
def righttest(xarr, yarr):
    '''Boolean array whose elements are True if x,y on right boundary of unit circle'''
    return np.logical_and(xarr >= 0, outsidecircle(xarr, yarr))
def bottomtest(xarr, yarr):
    '''Boolean array whose elements are True if x,y on bottom boundary of unit circle'''
    return np.logical_and(yarr < 0, outsidecircle(xarr, yarr))
def toptest(xarr, yarr):
    '''Boolean array whose elements are True if x,y on top boundary of unit circle'''
    return np.logical_and(yarr >= 0, outsidecircle(xarr, yarr))
def laplacian(field, deltax, deltay):
    '''approximate laplacian of 2D field using finite difference.
    deltax, deltay can be constant scalar or list of coordinates'''
    dudx, dudy = np.gradient(field, deltax, deltay)
    d2udx, null = np.gradient(dudx, deltax, deltay)
    null, d2udy = np.gradient(dudy, deltax, deltay)
    return d2udx + d2udy
def laplacian_NNNN(field, deltax, deltay):
    ''' ***NOT USED***
    approximate laplacian of 2D field using finite difference.
    deltax, deltay specify constant spacing of points'''
    # there is not any checking of the input array size
    # forward and backward difference for edge boundaries
    x0, x1, x2 = field[0,:], field[1,:], field[2,:]
    xb0, xb1, xb2 = field[-1,:], field[-2,:], field[-3,:]
    y0, y1, y2 = field[:,0], field[:,1], field[:,2]
    yb0, yb1, yb2 = field[:,-1], field[:,-2], field[:,-3]
    # central difference for middle portions
    x_A, x_B, x_C = field[0:-2,:], field[1:-1,:], field[2:,:]
    y_A, y_B, y_C = field[:,0:-2], field[:,1:-1], field[:,2:]
    # setup derivative arrays
    d2udx, d2udy = (np.zeros_like(field), np.zeros_like(field))
    # coefficients for 2nd derivative are 1, -2, 1 whether central, forward, or backward difference
    # x-direction
    d2udx[0,:] = (1*x0 - 2*x1 + 1*x2) / deltax**2
    d2udx[-1,:] = (1*xb0 - 2*xb1 + 1*xb2) / deltax**2
    d2udx[1:-1, :] = (1*x_A - 2*x_B + 1*x_C) / deltax**2
    # y-direction
    d2udy[:,0] = (1*y0 - 2*y1 + 1*y2) / deltay**2
    d2udy[:,-1] = (1*yb0 - 2*yb1 + 1*yb2) / deltay**2
    d2udy[:, 1:-1] = (1*y_A - 2*y_B + 1*y_C) / deltay**2
    return d2udx + d2udy
class heat_plot:
    '''helper for plotting'''
    def __init__(self, x, y, u, tmin, tmax):
        '''plot initial condition'''
        self.x = x
        self.y = y
        # colorbar scale limits
        self.tmin = tmin
        self.tmax = tmax
        fig = plt.figure(figsize=(7,6.4))
        # adjust size to fit long title
        fig.subplots_adjust(top=.82)
        plt.pcolormesh(x, y, u, cmap='rainbow', vmin=self.tmin, vmax=self.tmax)
        cbar = plt.colorbar(fraction=.08, shrink=.8)
        cbar.set_label('Temperature')
    def update(self, u, title):
        '''plot results'''
        plt.cla()
        plt.pcolormesh(self.x, self.y, u, cmap='rainbow', vmin=self.tmin, vmax=self.tmax)
        plt.title(title)
        plt.xlabel('x-coord')
        plt.ylabel('y-coord')
        plt.pause(0.05)
# setup initial conditions
# left boundary u=1, right boundary u=2
uleft = 1
uright = 2
u = np.ones_like(x) * 1.5
left = lefttest(x, y)
right = righttest(x, y)
u[left] = uleft
u[right] = uright
# tlow, thigh for colorbar scale limits
tlow, thigh = 0.8, 2.2
heatplot = heat_plot(x, y, u, tlow, thigh)
plt.show()
# iterate time steps
dt = 0.00005
niter = 801
plotinterval = 200
for i in range(niter+1):
    if i % plotinterval == 0:
        heatplot.update(u
,title = 
'''Heat Transfer for 2D Disk
Left-boundary u=1, Right-boundary u=2
Time, t = %.3f secs
Iteration Steps = %.0f'''
% (i*dt, i)
        ) # end update
    # end if
    ddt = laplacian(u, dx, dy)
    u = u + ddt*dt
    u[left] = uleft
    u[right] = uright
# end for
# plt.close()

# next scenario
# top half u=-1, bottom half u=1
utop = -1
ubottom = 1
u = np.ones_like(x) * 0
top = toptest(x, y)
bottom = bottomtest(x, y)
u[top] = utop
u[bottom] = ubottom
# tlow, thigh for colorbar scale limits
tlow, thigh = -1.2, 1.2
heatplot = heat_plot(x, y, u, tlow, thigh)
plt.show()
# iterate time steps
dt = 0.00005
niter = 801
plotinterval = 200
for i in range(niter+1):
    if i % plotinterval == 0:
        heatplot.update(u
,title = 
'''Heat Transfer for 2D Disk
Top-boundary u=-1, Bottom-boundary u=1
Time, t = %.3f secs
Iteration Steps = %.0f'''
% (i*dt, i)
        ) # end update
    # end if
    ddt = laplacian(u, dx, dy)
    u = u + ddt*dt
    u[top] = utop
    u[bottom] = ubottom
# end for
# plt.close()

# next scenario
# boundary fixed at u(𝜃)=cos(𝜃)
def cos_theta(x, y):
    '''u(𝜃)=cos(𝜃). Convert x, y to angle then compute cosine'''
    angle = np.angle(x + 1j*y)
    return np.cos(angle)
u = np.ones_like(x) * 0
outer = outsidecircle(x, y)
u[outer] = cos_theta(x[outer], y[outer])
# tlow, thigh for colorbar scale limits
tlow, thigh = -1.2, 1.2
heatplot = heat_plot(x, y, u, tlow, thigh)
plt.show()
# iterate time steps
dt = 0.00005
niter = 801
plotinterval = 200
for i in range(niter+1):
    if i % plotinterval == 0:
        heatplot.update(u
,title = 
'''Heat Transfer for 2D Disk
Boundary u(a)=cos(a)
Time, t = %.3f secs
Iteration Steps = %.0f'''
% (i*dt, i)
        ) # end update
    # end if
    ddt = laplacian(u, dx, dy)
    u = u + ddt*dt
    u[outer] = cos_theta(x[outer], y[outer])
# end for


######################
######################


import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
%matplotlib inline
# field equations
# v1 = 𝜋 sin 𝜋x cos 𝜋y
# v2 = -𝜋 cos 𝜋x sin 𝜋y
def vfunc(x, y):
    pi = np.pi
    v1 = -pi * np.sin(pi * x) * np.cos(pi * y)
    v2 = -pi * np.cos(pi * x) * np.sin(pi * y)
    return np.array((v1, v2))
# setup grid
xygrid = np.mgrid[0:2:16j, 0:1:16j]
x, y = xygrid
vgrid = vfunc(x, y)
v1, v2 = vgrid
# magnitude for color scale
vmag = np.sqrt(v1 ** 2 + v2 ** 2)
fig, axs = plt.subplots(figsize=(7,7))
# avoids divide by zero error
magdivisor = np.where(vmag>0, vmag, 1)
# plot field of arrows
plt.quiver(x, y, v1 / magdivisor, v2 / magdivisor, vmag, cmap='cool')
cbar = plt.colorbar(orientation='horizontal', fraction=.04)
cbar.set_label('V magnitude')
plt.title('Vector Field and Trajectories')
plt.xlabel('x-position'), plt.ylabel('y-position')
# trajectory for x=0 to 2, y=0.5
# for time t=0 to 10
xyinit = np.mgrid[0:2:10j, .5:1:1j]
length = xyinit[0].size
xslice = slice(0, length)
yslice = slice(length, None)
tend = 10
t = np.arange(0, tend, .01)
# vector unravelled for integrator
sol = solve_ivp(
    lambda t, arr: vfunc(arr[xslice], arr[yslice]).ravel(),
    t_span = (0, tend),
    y0 = xyinit.ravel(),
    t_eval = t,
    atol = 1e-9,
    rtol = 1e-9
)
# reshape to columns of paths
nt = sol.y.shape[1]
xsolution = sol.y[xslice,:].transpose()
ysolution = sol.y[yslice,:].transpose()
# color cycle
# axs[0].set_prop_cycle(color=mp.cm.ScalarMappable(cmap='jet').to_rgba(range(10)))
# plot trajectories
plt.plot(xsolution, ysolution, linewidth=2, color='blue')
# plt.grid()
plt.title('Vector Field and Trajectories')
plt.xlabel('x-position'), plt.ylabel('y-position')
# plt.xlim(0, 2)
# plt.ylim(0, 1)
plt.show()



