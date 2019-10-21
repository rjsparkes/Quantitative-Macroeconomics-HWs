#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:27:25 2019

@author: abhinav,giada,richard
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


# Paremeters
b=0.988   #beta
th=.679 #theta
d=.013  #delta
kmax = 5.24 #kappa
sig = 2 # consumption elasticity 


# Steady state capital
ks=(1/(1-th)*((1/b)+d-1))**(-1/th) 

# k and h grid 
k=np.linspace(0.01,2*ks,100) #close to steady state
h=np.linspace(0,1,5)  
x,y,z=np.meshgrid(k,k,h)

# Value function
V = np.empty(shape=[100, 450])
V[:,0]=np.zeros((100))  # initial guess of 0 

# Defining return matrix including h
def return_mat(k1,k2,h1):
    return k1**(1-th)*h1**(th) + (1-d)*k1 - k2
N = return_mat(x,y,z)

# Utility function
def utility(c1,c2,h1):
    for i in range(400):
        for i,j,m in zip(range(0,100),range(0,100),range(0,5)):
                if N[i,j,m]>=0:
                    return np.log(c1**(1-th)*h1**(th) + (1-d)*c1 - c2) - kmax*((h1**(1+(1/sig)))/(1+(1/sig)))
            
M = utility(x,y,z)
M[np.isnan(M)] = 0

## part a: Brute force
 
# Starting time
start= timer()

# Value function iterations
X = np.empty(shape=[100, 400,10])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            for m in range(5):
                X[i,j,m]=M[i,j,m]+(b*V[:,n][j])        
    for i in range(0,100):
        V[:,n+1][i]= np.amax(X[:,i]) 
        P[:,n][i]=np.argmax(X[:,i]) 
        for i in range(0,100):
            if abs(V[:,n+1][i]-V[:,n][i])> err:
                continue
            else:
                iter +=1
                break
     

#end time
end= timer()
iter_time= end - start
print('Brute force time', iter_time)
print('Brute force iteration',iter) 

#plot
fig1=plt.figure(figsize=(8,6))
plt.plot(k,V[:,400],color='blue')
plt.title('Figure 2:Brute force  Labor supply value Function ')
plt.xlabel('Capital')

## part b: Monotonocity
 
# Starting time
start= timer()

# Value function iterations
X = np.empty(shape=[100, 400,5])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            for m in range(5):
                if j >= P[:,n+1][i]:  # Monotonicity of k
                    if m >= P[:,n+1][i]: # Monotonicity of h 
                     X[i,j,m]=M[i,j,m]+(b*V[:,n][j])        
    for i in range(0,100):
        V[:,n+1][i]= np.amax(X[:,i]) 
        P[:,n][i]=np.argmax(X[:,i]) 
        for i in range(0,100):
            if abs(V[:,n+1][i]-V[:,n][i])> err:
                continue
            else:
                iter +=1
                break
     

#end time
end= timer()
iter_time= end - start
print('Monotonicity force time', iter_time)
print('Monotonicity iteration',iter) 


## part c: Concavity
 
# Starting time
start= timer()

# Value function iterations
V1 = np.empty(shape=[100, 450])
X1 = np.empty(shape=[100, 400,5])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            for m in range(5):
                if X1[i,j-1,m-1]<X1[i,j,m]: # Concavity condition
                    X1[i,j,m]=M[i,j,m]+(b*V[:,n][j])        
    for i in range(0,100):
        V1[:,n+1][i]= np.amax(X1[:,i]) 
        P[:,n][i]=np.argmax(X1[:,i]) 
        for i in range(0,100):
            if abs(V1[:,n+1][i]-V1[:,n][i])> err:
                continue
            else:
                iter +=1
                break
     

#end time
end= timer()
iter_time= end - start
print('Concavity force time', iter_time)
print('Concavity iteration',iter) 


## part d: Local search
 
# Starting time
start= timer()

# Value function iterations
X = np.empty(shape=[100, 400,5])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            for m in range(5):
                if j == P[:,n][i]:
                    if m== P[:,n][i]:
                     X1[i,j,m]=M[i,j,m]+(b*V[:,n][j]) # Local search 
    for i in range(0,5):
        V1[:,n+1][i]= np.amax(X[:,i]) 
        P[:,n][i]=np.argmax(X[:,i]) 
        for i in range(0,100):
            if abs(V1[:,n+1][i]-V1[:,n][i])> err:
                continue
            else:
                iter +=1
                break
     

#end time
end= timer()
iter_time= end - start
print('Local search force time', iter_time)
print('Local search iteration',iter) 

## part e: Monotonicity and concavity 

# Starting time
start= timer()

# Value function iterations
V1 = np.empty(shape=[100, 450])
X1 = np.empty(shape=[100, 400,5])
X = np.empty(shape=[100, 400,5])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            for m in range(5):
                if j >= P[:,n+1][i]: # monotonicity of k 
                    if m >= P[:,n+1][i]: # monotonicity of h
                        if X1[i,j-1,m-1]<X1[i,j,m]: # concavity
                            X1[i,j,m]=M[i,j,m]+(b*V[:,n][j])
                            for i in range(0,5):
                                V1[:,n+1][i]= np.amax(X1[:,i]) 
                                P[:,n][i]=np.argmax(X1[:,i])
                                for i in range(0,100):
                                    if abs(V1[:,n+1][i]-V1[:,n][i])> err:
                                        continue
                                    else:
                                        iter +=1
                                        break
#end time
end= timer()
iter_time= end - start
print('Monotonicity and concavity time', iter_time)
print('Monotonicity and concavity iteration',iter) 


        
