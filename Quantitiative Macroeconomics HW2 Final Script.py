#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:38:21 2019

@author: giadabozzelli
"""

#Import lybraries 
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the CES function we want to approximate 
def f(k,h):
  return ((1-alpha)*k**((sigma-1)/sigma)+alpha*h**((sigma-1)/sigma))**(sigma/(sigma-1))

#Define parameters
alpha = 0.5 
sigma = 0.25 #5.00 and 1.0000001
k_lims=[0,10]
h_lims=[0,10]
n= 20 #nodes
a= 0 
b= 10

## Step 1: Defining chebyshev nodes
def Chebyshev_nodes(a, b, n):
   from math import cos, pi
   cheby_nodes = [0.5*(a+b) + 0.5*(b-a)*cos(float(2*i+1)/(2*(n+1))*pi)
            for i in range(n+1)]
   return cheby_nodes

nodes= Chebyshev_nodes(a, b, n)
knodes= nodes
hnodes= nodes
k1=np.array(knodes)
h1=np.array(hnodes)
K,H = np.meshgrid(k1, h1)
w = f(k1[:,None],h1[None,:])
w = np.matrix(w)

#Step 2: compute Chebyshev polynomials and coefficients
def cheby_polynomial(d,x):  #x=nodes
	pi = []
	pi.append(np.ones(len(x)))
	pi.append(x)
	for i in range (1,d):
		p = 2 * x * pi[i-1] - pi[i-2]
		pi.append(p)
	pi_chebyshev = np.matrix(pi[d])
	return pi_chebyshev

def cheby_coeff(y,w,d):
  beta = np.empty((d+1)*(d+1))
  beta.shape = (d+1,d+1)
  for i in range(d+1):
      for j in range(d+1):
          beta[i,j] = (np.sum(np.array(w)*np.array((np.dot(cheby_polynomial(i,x).T,cheby_polynomial(j,x)))))
                        /np.array((cheby_polynomial(i,x)*cheby_polynomial(i,x).T)*(cheby_polynomial(j,x)*cheby_polynomial(j,x).T)))
  return beta


#Step 3: approximate the function with Chebyshev polynomials and coefficients
def cheby_approx(x,y,beta,d):
  g = []
  val1 = ((2*((x-a)/(b-a)))-1)
  val2 = ((2*((y-a)/(b-a)))-1)
  for i in range(d):
      for j in range(d):
              g.append(np.array(beta[i,j])*np.array((np.dot(cheby_polynomial(i,val1).T,cheby_polynomial(j,val2)))))
  g_sum = sum(g)
  return g_sum

#Approximation for degree = 3

d = 3
x = k1
y = h1

beta = cheby_coeff(nodes,w,d)
approx = cheby_approx(x,y,beta,d)
K,H= np.meshgrid(k1, h1)
real = f(K, H)
error=abs(real-approx)

#Setting 3 different subplots
fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('Graphs: σ=1.0000001, Order 3', fontsize=16)

ax = fig.add_subplot(131, projection='3d')
ax.set_title("Exact CES function order 3")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, real, cmap=cm.inferno, linewidth=0, antialiased=False)
ax.view_init(azim=200)

ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated CES function order 3")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, approx, cmap=cm.inferno, linewidth=0, antialiased=False)
ax.view_init(azim=200)


ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated errors order 3")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, error, cmap=cm.inferno, linewidth=0, antialiased=False)
ax.view_init(azim=200)

plt.show()

# Approximation for degree = 7

d = 7
x = k1
y = h1


#Setting 3 different subplots
fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('Graphs: σ=1.0000001, Order 7', fontsize=16)

ax = fig.add_subplot(131, projection='3d')
ax.set_title("Exact CES function order 7")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, real, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated CES function order 7")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, approx, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)


ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated errors order 7")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, error, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=200)

plt.show()

# Approximation for degree = 11

d = 11
x = k1
y = h1



#Setting 3 different subplots
fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('Graphs: σ=1.0000001, Order 11', fontsize=16)

ax = fig.add_subplot(131, projection='3d')
ax.set_title("Exact CES function order 11")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, real, cmap=cm.magma, linewidth=0, antialiased=False)
ax.view_init(azim=200)

ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated CES function order 11")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, approx, cmap=cm.magma, linewidth=0, antialiased=False)
ax.view_init(azim=200)


ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated errors order 11")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, error, cmap=cm.magma, linewidth=0, antialiased=False)
ax.view_init(azim=200)

plt.show()

# Approximation for degree = 15

d = 15
x = k1
y = h1


#Setting 3 different subplots
fig = plt.figure(figsize=plt.figaspect(0.25))
fig.suptitle('Graphs: σ=1.0000001, Order 15', fontsize=16)

ax = fig.add_subplot(131, projection='3d')
ax.set_title("Exact CES function order 15")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, real, cmap=cm.plasma, linewidth=0, antialiased=False)
ax.view_init(azim=200)

ax = fig.add_subplot(132, projection='3d')
ax.set_title("Approximated CES function order 15")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, approx, cmap=cm.plasma, linewidth=0, antialiased=False)
ax.view_init(azim=200)


ax = fig.add_subplot(133, projection='3d')
ax.set_title("Approximated errors order 15")
K,H= np.meshgrid(k1, h1)
ax.set_xlabel('K')
ax.set_ylabel('H')
ax.set_zlabel('f(k,h)')
ax.plot_surface(K, H, error, cmap=cm.plasma, linewidth=0, antialiased=False)
ax.view_init(azim=200)

plt.show()


#ISOQUANTS

iso=[5,10,25,50,75,90,95]
iso1=[]
for i in iso:
    iso_percent=np.percentile(real,i)
    iso1.append(iso_percent)

iso2=[]
for i in iso:
    iso_percent2=np.percentile(approx,i)
    iso2.append(iso_percent)

#Plot isoquants   
fig = plt.figure(figsize=(8.5,10))
ax = plt.subplot(311)
ax.set(title='Isoquant True Plot')
cs = ax.contour(K,H,real, iso1)
plt.clabel(cs, fontsize=10, inline=1)
plt.xlabel('k')
plt.ylabel('h')
plt.xlim(0,10)
plt.ylim(0,10)

ax2 = plt.subplot(312,sharex=ax, sharey=ax)
ax2.set(title='Isoquant Approx Plot')
ax2.contour(K,H,approx, iso1)
plt.xlabel('k')
plt.ylabel('h')
plt.xlim(0,10)
plt.ylim(0,10)

plt.show()

