#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns; sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt
import random 
import pandas as pd

## Ex 1.1
# Defining joint normal distribution
cov= [[1,0],[0,1]]  # covariance of 0 and variance 1 
# cov= [[1,0.50],[0.50,1]] covariance of -0.50
# cov= [[1,-0.50],[-0.50,1]] covariance of -0.50
mu=[1,1] # assuming mean of zero for z
data=np.random.multivariate_normal(mean=mu,cov=cov,size=10000000)
lnki = data[:,0]
lnzi = data[:,1]

# Plotting joint density
# Logs
with sns.axes_style("white"):
    sns.jointplot(x=lnki, y=lnzi, kind="hex", color="k");
# levels
ki= np.exp(lnki)
zi= np.exp(lnzi)
with sns.axes_style("white"):
    sns.jointplot(x=ki, y=zi, kind="hex", color="k"); 
    
## Ex 1.2
# Defining output for each firm
gamma=0.6
# gamma=0.8 Ex 2
yi=zi*ki**(gamma)

## Ex 1.3
K=sum(ki)
n=10000000
z=np.empty(n)
k_eff=np.empty(n)

for i in range(n):
    z[i]= (zi[0]/zi[i])**(1/(gamma-1))
k_eff[0] = K/sum(z)
k_eff = z*k_eff[0]

## Ex 1.4
diff=ki-k_eff
plt.plot(diff)
plt.title('Comparing effecient and actual capital')

## Ex 1.5
ye=zi*k_eff**(gamma)
ye=sum(ye)
yi=sum(yi)
output_gain= ((ye/yi)-1)*100
print('Output gain from reallocation',output_gain)
    
## EX 3.1
# Random sampling 
n1=10000

## Ex 3.5 redoing 1-4 using different sample size
# n1=100
# n1=1000
# n1=100000

lnk=list(lnki)
lnz=list(lnzi)
lnki1=random.sample(lnk,n1)       
lnzi1=random.sample(lnz,n1)  

# Checking variance and covariance for sample
lnki1_var=np.var(lnki1)
lnzi1_var=np.var(lnzi1)
cov_random=np.cov(lnki1,lnzi1)
print('Variance capital sample',lnki1_var)
print('Variance productivity sample',lnzi1_var)
print('Covariance sample',cov_random)

## Ex 3.2
# Redoing items 3-5
#3
ki1= np.exp(lnki1)
zi1= np.exp(lnzi1)
K1=sum(ki1)
z1=np.empty(n1)
k_eff1=np.empty(n1)

for i in range(n1):
    z1[i]= (zi1[0]/zi1[i])**(1/(gamma-1))
k_eff1[0] = K1/sum(z1)
k_eff1 = z1*k_eff1[0]

#4
diff1=ki1-k_eff1
plt.plot(diff1)
plt.title('Comparing effecient and actual output:sample data')
plt.show()

diff=ki-k_eff
plt.plot(diff)
plt.title('Comparing effecient and actual capital')
plt.show()

#5
ye1=zi1*k_eff1**(gamma)
ye1=sum(ye1)
yi1=zi1*ki1**(gamma)
yi1=sum(yi1)
output_gain1= ((ye1/yi1)-1)*100
print('Output gain from reallocation sample',output_gain1)
    
# Comparing output gain
gain_diff=output_gain1-output_gain
print('Comparison of output gain for population and sample',gain_diff)


## Ex 3.3
output_gain2=np.empty(1000)

for i in range(1000):
    # Random sampling
    lnki2=random.sample(lnk,n1)       
    lnzi2=random.sample(lnz,n1)
    # Optimizing
    ki2= np.exp(lnki2)
    zi2= np.exp(lnzi2)
    K2=sum(ki2)
    z2=np.empty(n1)
    k_eff2=np.empty(n1)
    for j in range(n1):
        z2[j]= (zi2[0]/zi2[j])**(1/(gamma-1))
    k_eff2[0] = K2/sum(z2)
    k_eff2 = z2*k_eff2[0]
    # optimal gain
    ye2=zi2*k_eff2**(gamma)
    ye2=sum(ye2)
    yi2=zi2*ki2**(gamma)
    yi2=sum(yi2)
    output_gain2[i]= ((ye2/yi2)-1)*100

# plotting and stats of distribution of output gain
plt.hist(output_gain2)
plt.title('Distribution of output gain')
plt.show()
df = pd.DataFrame(output_gain2)
stats = df.describe()
print(stats)

## Ex 3.4
u_bound=output_gain*1.1
l_bound=output_gain*0.9

probs=[]
for i in range(1000):
    if (output_gain2[i]>=l_bound) and (output_gain2[i]<=u_bound):
        prob=output_gain2[i]
        probs.append(prob)

totprob=len(probs) 
prob1= (totprob/1000)*100
print('probabality of sample in 10% interval',prob1)     
        






