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

# k grid 
k=np.linspace(0.01,2*ks,100) #close to steady state
x,y=np.meshgrid(k,k)

# Value function
V = np.empty(shape=[100, 450])
V[:,0]=np.zeros((100))  # initial guess of 0 

# Defining return matrix
def return_mat(k1,k2):
    return k1**(1-th) + (1-d)*k1 - k2
N = return_mat(x,y)

# Utility function
def utility(c1,c2):
    for i in range(400):
        for j in range (100):
                if N[i,j]>=0:
                    return np.log10(c1**(1-th) + (1-d)*c1 - c2) - (kmax/(1 + 1/sig))
            
M = utility(x,y)
M[np.isnan(M)] = -100

## part a: Brute force
 
# Starting time
start= timer()

# Value function iterations
X = np.empty(shape=[100, 400])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
                X[i,j]=M[i,j]+(b*V[:,n][j])        
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
plt.plot(k,V[:,400],color='orange')
plt.title('Figure 1:Brute force value Function')
plt.xlabel('Capital')

## part b: Monotonicity 

# Starting time
start= timer()

# Value function iterations
X = np.empty(shape=[100, 400])
P= np.empty(shape=[100, 450]) #policy function
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            if j >= P[:,n+1][i]: # Adding monotoncity constraint
                X[i,j]=M[i,j]+(b*V[:,n][j])        
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
print(' Monotonicity time', iter_time)
print('Monotonicity iteration',iter) 

## part c: Concavitity

# Starting time
start= timer()

# Value function iterations
V1 = np.empty(shape=[100, 450])
X1 = np.empty(shape=[100, 400])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            if X1[i,j-1]<X1[i,j]: #Concavity constraint
                X1[i,j]=M[i,j]+(b*V[:,n][j])        
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
print('Concavity time', iter_time)
print('Cocavity iteration',iter) 

## part d: Local search

# Starting time
start= timer()

# Value function iterations
X = np.empty(shape=[100, 400,4])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            if j == P[:,n][i]: # Local search
                X1[i,j]=M[i,j]+(b*V[:,n][j])        
    for i in range(0,100):
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
print('Local search time', iter_time)
print('Local search iteration',iter) 


## part e: Monotonicity and concavity 

# Starting time
start= timer()

# Value function iterations
V1 = np.empty(shape=[100, 450])
X1 = np.empty(shape=[100, 400])
X = np.empty(shape=[100, 400])
P= np.empty(shape=[100, 450])
iter=0 # starting iteration
for n in range(0,400):
    err=0.01
    for i in range(100):
        for j in range(100):
            if j >= P[:,n+1][i]:#Monotonicity
              if X1[i,j-1]<X1[i,j]: #Concavity
                X1[i,j]=M[i,j]+(b*V[:,n][j])        
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
print('Monotonicity and concavity time', iter_time)
print('Monotonicity and concavity iteration',iter) 
