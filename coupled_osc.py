#!/usr/bin/python

import math
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as linalg 
import matplotlib.animation as animation
from scipy.integrate import odeint
import sys

k = 3.0
m = 1.0
n = 6
spacing = 3.0
mode = 2 # no greater than n
w0_2 = k/m
W = np.zeros(shape = (n,n))	#mass matrix
D = [spacing*i for i in range(0,n+2)]

X = []
inits = []

for i in range(0, n):
	x0 = float(input("x0 for oscillator " + str(i) + ": "))
	v0 = float(input("v0 for oscillator " + str(i) + ": "))
	inits += [x0, v0]

for i in range(0, n):
	if not i == 0:
		W[i,i-1] = w0_2
	W[i,i] = -2*w0_2
	if not i == n - 1:
		W[i,i+1] = w0_2

# for i in range(0, n):

def derivs(w, t, radius = 0.05):
	f = []
	yprime = 0.0
	for i in range(0, n):
		f.append(w[2*i+1])
		if i == 0:
			yprime = W[i,i]*w[i] + W[i, i+1]*w[2*(i+1)] #- 0.3*w[2*i+1]**2
		elif i == n-1:
			yprime = W[i,i-1]*w[2*(i-1)] + W[i,i]*w[2*i] #- 0.3*w[2*i+1]**2
		else:
			yprime = W[i, i-1]*w[2*(i-1)] + W[i,i]*w[2*i] + W[i,i+1]*w[2*(i+1)] #- 0.3*w[2*i+1]**2
		f.append(yprime)

	return f

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 100.0
dt = 0.05

t = np.arange(0.0, stoptime, dt)

# Call the ODE solver.
wsol = odeint(derivs, inits, t,
              atol=abserr, rtol=relerr)
print wsol
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(D[0], D[-1]), ylim=(-6, 6))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [wsol[i][2*j] for j in range(0, n)]
    
    for i in range(1,len(D)-1):
    	thisx[i-1] += D[i]

    thisy = [0.0 for j in range(0, n+2)]
    thisx = [D[0]] + thisx + [D[-1]]
 
    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(wsol)),
    interval=25, blit=True, init_func=init)

plt.show()

for i in range(0,n):
	fig = plt.figure(i)
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(xs=[x[2*i] for x in wsol], ys=[m*x[2*i+1] for x in wsol], zs=t)
plt.draw()
plt.show()
