#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:27:00 2019

@author: abhinav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:39:16 2019

@author: giadabozzelli, abhinav and richard
"""


import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy.polynomial.chebyshev as poly

# Paremeters
b=0.988   #beta
th=.679 #theta
d=.013  #delta
kmax = 5.24 #kappa
sig = 2 # consumption elasticity 
h=1
n=10
ss= 2

# Steady state capital
ks=(1/(1-th)*((1/b)+d-1))**(-1/th) 

# k grid 
k=np.linspace(0.01,ss*ks,n) 
x,y=np.meshgrid(k,k)

# Chebyshev nodes
xn= []
for i in range(n):
   x = np.cos((((2*i)-1)/200)*np.pi)
   xn.append(x)
  
xn= np.asarray(xn)

xn1= []
for i in range(n):
   x = ((xn[i])*((ss*ks - k[0])/2))+ (((ss*ks + k[0])/2))
   xn1.append(x)
  
xn1= np.asarray(xn1)


# Value function
y_init=np.ones(n)
p_poly=poly.chebfit(xn1,y_init,2) 
V0=poly.chebval(k,p_poly) 

# Defining return matrix
def return_mat(k1,k2):
    return k1**(1-th) + (1-d)*k1 - k2
N = return_mat(x,y)

# Utility function
def utility(c1,c2):
    for i in range(0,n):
        for j in range (0,n):
                if N[i,j]>=0:
                    return np.log(c1**(1-th)*h**(th) + (1-d)*c1 - c2) - kmax*((h**(1+(1/sig)))/(1+(1/sig)))
                else:
                    return -1000
            
M = utility(x,y)

# Defining policy functions 
X = np.empty(shape=[n, n])
P= np.empty(shape=[n, 1])

for i in range(n):
    for j in range(n):
     X[i,j]=M[i,j]+(b*V0[j])
     X[np.isnan(X)] = -10
for i in range(n):
    for j in range(n): 
     P[i]=np.argmax(X[:,i])
        
            
# Updating value function                    
y1 = np.empty(shape=[n, 1])   
for i in range(0,n):
    y1[i]=utility(k[i],P[i])+(b*V0[i])
y1[np.isnan(y1)] = 0
y1=np.reshape(y1, (n,))

# Calculation new chebyshev polynomial
p_poly1=poly.chebfit(xn1,y1,2)     
diff=np.amax(p_poly - p_poly1)

# Starting time
start= timer()

# Value function iterations
iter=0 # starting iteration
err=0.01
maxiter=1000
while diff>0.01 and iter<maxiter:
    V1=poly.chebval(k,p_poly1)
    for i in range(0,n):
        for j in range(0,n):
         X[i,j]=M[i,j]+(b*V1[j]) 
         X[np.isnan(X)] = -10 
    for i in range(0,n):
       P[i]=np.argmax(X[i,:]) 
       for i in range(0,n):
            y1[i]=utility(k[i],P[i])+(b*V1[i])
            y1[np.isnan(y1)] = -10
            y1=np.reshape(y1, (n,))
    p_poly1=poly.chebfit(xn1,y1,2)
    diff=np.amax(abs(p_poly - p_poly1))
    iter +=1
   


#end time
end= timer()
iter_time= end - start
print('Brute force Chebyshev time', iter_time)
print('Chebyshev iteration',iter) 

#plot
fig3=plt.figure(figsize=(8,6))
plt.plot(k,V1,color='blue')
plt.title('Figure 3:Brute force Chebyshev value Function ')
plt.xlabel('Capital')
plt.plot(k,V1)


