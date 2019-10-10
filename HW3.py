import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

"""
Created on Mon Oct  7 19:03:00 2019

@author: abhinav,giada,richard 
"""

## Question 1 

## Part A : steady state
theta = 0.67
h = 0.31
delta = 0.0625

beta = 1/((1-theta)/4 +1 - delta)
z = ((1/beta - 1 + delta)/(1-theta))**((1-theta)/theta)*(1/h)
k = ((1-beta*(1-delta))/(beta*(1-theta)*(z*h)**theta))**(-1/theta)
y = k**(1 - theta)*(z*h)**theta
c = y - delta*k
i = delta*k

# Values at steady state 
print('Z Steady State = '"%.2f" % z) 
print('Capital Steady State = '"%.2f" % k) 
print('Output Steady State = '"%.2f" % y) 
print('Consumption Steady State = '"%.2f" % c) 
print('Investment Steady State = '"%.2f" % i) 

## Part B: New steay state 
theta = 0.67
h = 0.31
delta = 0.0625

beta = 1/((1-theta)/4 +1 - delta)
z1= z*2 
k1 = ((1-beta*(1-delta))/(beta*(1-theta)*(z1*h)**theta))**(-1/theta)
y1 = k1**(1 - theta)*(z1*h)**theta
c1 = y1 - delta*k1
i1 = delta*k1

# Values at steady state
print('Z New Steady State = '"%.2f" % z1) 
print('Capital New Steady State = '"%.2f" % k1) 
print('Output New  Steady State = '"%.2f" % y1) 
print('Consumption New Steady State = '"%.2f" % c1) 
print('Investment New Steady State = '"%.2f" % i1) 


## Part C: transition from steady state to new steady state

# For this part we use the ramsay model 
# this code has been found by a free online resource and we have modified it for our case.
class ramseyModel(object):
    def __init__(self, params, k=None, c=None):
           "A discrete time version of the Ramsey-Cass-Koopmans model."
           """ Initializes a ramseyModel object.        
           Attributes: 
        
            1. params: a dictionary of parameters and their values.
            2. k: an initial condition for the state variable k, capital per 
               effective worker.
            3. c: an initial condition for the control variable c, consumption 
               per effective worker.
        
        """
        # current value of state variable, k
           self.k            = k
        # current value of the control variable, c
           self.c            = c
        # dictionary of parameter values
           self.param_dict   = params
        # dictionary of steady state values        
           self.SS_dict      = {'k_star':self.set_k_star(self.param_dict), 
                                'c_star':self.set_c_star(self.param_dict),
                                's_star':self.set_s_star(self.param_dict)}

    def set_k_star(self,params): 
        "The steady-state level of capital stock per effective worker"
       
        # extract params
        delta = params['delta']
        theta = params['theta']
        beta = params['beta']
        h = params['h']
        z = params['z']
        v = params['v'] # we keep this paremeter for later evalution with taxes
            
        return pow(((1-theta)/((1/beta)+delta+v-1)), (1/theta))*(z*h)    

    def set_c_star(self,params): 
        "The steady-state level of consumption per effective worker"
        
        # extract params
        delta = params['delta']
        theta = params['theta']
        beta = params['beta']
        h = params['h']
        z = params['z']
        v = params['v']
        k_star = self.set_k_star(params)
        
        return pow(k_star, 1-theta)*pow(z*h, theta) - delta*k_star

    def set_s_star(self,params):
        "Steady state savings rate of the Ramsey economy"
        
        # extract params
        delta = params['delta']
        
        return delta      

    def capital(self, k, c):
        "Next period's capital stock per effective worker"

        # extract params
        theta = self.param_dict['theta']
        delta = self.param_dict['delta']
        h = self.param_dict['h']
        z = self.param_dict['z']
        v = self.param_dict['v']
    
        return pow(k, 1-theta)*pow(z*h, theta) - c + (1-delta-v)*k

    def euler(self, k, c):        
        """Via the consumption Euler equation, next period's consumption per
        effective worker can be written as a function of current period 
        consumption and capital stock)."""
        h = self.param_dict['h']
        z = self.param_dict['z']
        delta = self.param_dict['delta']
        theta = self.param_dict['theta']
        beta = self.param_dict['beta']
        v = self.param_dict['v']
        return c*beta*(1 - delta - v + (1-theta)*pow(z*h, theta)*pow(k, -theta))
    
    def update(self):
        """Update the state variables according to: 
        kplus = capital(k, c)
        cplus = euler(k, c) """
    
        self.k = self.capital(self.k, self.c) 
        self.c = self.euler(self.k, self.c)

    def sample_path(self, N=None):
        """Generates a sample path of the Ramsey economy of length N starting 
        from the current state."""
        path = np.zeros(shape=(N, 2))
        
        for t in range(N):
            path[t, 0] = self.k
            path[t, 1] = self.c
            self.update()
        
        return path

    def get_numericSteadyState(self, k0=None, c0=None):
        """Finds the steady state for the Ramsey economy using fsolve."""
        # function to be optimized at steady state
        def ramseySS(x):
            out = [self.capital(x[0], x[1]) - x[0]]
            out.append(self.euler(x[0], x[1]) - x[1])
            return out
        x=optimize.fsolve(func=ramseySS, x0=(k0, c0))
        return optimize.fsolve(func=ramseySS, x0=(k0, c0))


    def forward_shoot(self, k0=None, c0=None, tol=1.5e-08):
        """Computes the full, non-linear saddle path for the Ramsey model"""
        
         # extract params
        theta = self.param_dict['theta']
        h = self.param_dict['h']
        z = self.param_dict['z']
        
        # compute steady state values
        k_star, c_star = self.SS_dict['k_star'], self.SS_dict['c_star']
        
        if k0 <= k_star:
            c_l = 0
            c_h = c_star
        else:
            c_l = c_star
            c_h =pow(k0, 1-theta)*pow(z*h, theta) 
        c0 = (c_h + c_l) / 2
        self.k, self.c = k0, c0
    
        # Initialize a counter
        count  = 0
        N_iter = 0
        
        # Forward Shooting Algorithm
        while 1:
            self.update()
            dist = np.abs(((self.k - k_star)**2 + (self.c - c_star)**2)**0.5)
            count = count + 1
            if k0 <= k_star:
                if self.k > k_star:
                    if dist < tol:
                        break
                    else: 
                        c_l = c0
                        c0 = (c_h + c_l) / 2
                        self.k, self.c = k0, c0
                        count = 0
                if self.c > c_star:
                    if dist < tol:
                        break
                    else:
                        c_h = c0 
                        c0 = (c_h + c_l) / 2
                        self.k, self.c = k0, c0
                        count = 0
            else:
                if self.k < k_star:
                    if dist < tol:
                        break
                    else: 
                        c_h = c0 
                        c0 = (c_h + c_l) / 2
                        self.k, self.c = k0, c0
                        count = 0
                if self.c < c_star:
                    if dist < tol:
                        break
                    else: 
                        c_l = c0
                        c0 = (c_h + c_l) / 2
                        self.k, self.c = k0, c0
                        count = 0
                
        self.k, self.c = k0, c0
        solutionPath = self.sample_path(count)

        return [self.c, solutionPath, count, dist]
    
# Setting paremeters at steady state
params = {'theta':0.67, 'delta':0.0625, 'h':0.31, 'beta':0.9804, 'z':1.63, 'v':0}

# Create an instance of the class ramseyDS
ramsey = ramseyModel(params)
solution=ramsey.get_numericSteadyState(k0=3, c0=0.8)
k_star=solution[0]
c_star=solution[1]
y_star=0.0625*k_star+c_star
v=k_star/y_star
i=0.0625*k_star/y_star

# settting new paremeters
params = {'theta':0.67, 'delta':0.0625, 'h':0.31, 'beta':0.9804, 'z':3.26, 'v':0}

# modelling for transition
ramsey = ramseyModel(params)
transition = ramsey.forward_shoot(k0=4,c0=0.75, tol=1.5e-4)
theta=0.67
delta=0.0625
z=1.63
h=0.31
beta=0.9804
k_initial=4.00
c_initial=0.75
k_star=8.00
c_star=1.50

# Using locus command for graphing 
def locusK(k):
    return pow(k, 1-theta)*pow(z*h, theta) - delta*k

# defining dimensions
gridmax, gridsize = 200, 10000
grid = np.linspace(0, gridmax, gridsize)

#plotting
fig1=plt.figure(figsize=(8,6))
plt.plot(grid, locusK(grid), '-', color='yellow', label=r'$deltak$')
z=3.25
plt.plot(grid, locusK(grid), '-', color='green', label=r'$deltak1$')
plt.axvline(k_initial, color='blue', label=r'$delta c$')
plt.plot(k_initial, c_initial, marker='.', markersize=12, color='k')
plt.axvline(k_star, color='purple', label=r'$delta c1$')
plt.plot(k_star, c_star, marker='.', markersize=12, color='k')
plt.plot(k_transition, c_transition, color='orange', label='Transition')
plt.ylim(0, 2.5)
plt.xlim(-5, 70)
plt.xlabel('Capital')
plt.ylabel('Consumption')
plt.legend(frameon=False)
plt.title('Figure 1: Transition graph', fontsize=16)

## part d: shock after 10 periods of steady state

# Setting for steady state 
params = {'theta':0.67, 'delta':0.0625, 'h':0.31, 'beta':0.9804, 'z':1.625, 'v':0}
ramsey= ramseyModel(params)
k_star=4.00
c_star=0.75
k_star, c_star = ramsey.SS_dict['k_star'], ramsey.SS_dict['c_star']
ramsey.k, ramsey.c = k_star, c_star

# 10 periods of steady state until shock
kc = [(k_star, c_star)]
Pathkc= np.repeat(kc,10)
Pathy = np.repeat(y_star,10)

# Create a grid of points for plotting
gridmax, gridsize = 200, 10000
grid = np.linspace(0, gridmax, gridsize)

# Create a new plot
plt.figure(figsize=(8,8))

# Adding  c and k locii
plt.plot(grid, locusK(grid), color='orange', linestyle='dashed', label=r'$\Delta k_{old}$')
plt.axvline(ramsey.SS_dict['k_star'], linestyle='dashed', color='k', label=r'$\Delta c_{old}$')
plt.plot(ramsey.SS_dict['k_star'], ramsey.SS_dict['c_star'], marker='.', markersize=10, color='k')

# Labelling axis
plt.xlim(0, 25)
plt.xlabel('k')
plt.ylim(0, 5)
plt.ylabel('c', rotation='horizontal')

# shock z! 
new_params = params
z2 = 2*z
new_params['z'] = z2
ramsey_new = ramseyModel(new_params)

# Add the c and k locii
plt.plot(grid, locusK(grid), color='orange', label=r'$\Delta k_{new}$')
plt.axvline(ramsey_new.SS_dict['k_star'], linestyle='solid', color='k', label=r'$\Delta c_{new}$')
plt.plot(ramsey_new.SS_dict['k_star'], ramsey_new.SS_dict['c_star'], marker='.', markersize=10, color='k')

# solve for the non-linear saddle path
ramsey_solution = ramsey_new.forward_shoot(k_star, tol=1.5e-4)

# Plot the full saddle-path
plt.plot(ramsey_solution[1][:, 0], ramsey_solution[1][:, 1], color='r', label='Saddle path')
plt.title('Figure 2: Effect of shock to z')
plt.legend(loc='best', frameon=False)
plt.show()


## Part E2: introduction of capital tax in old steady state

# we assume tax is 4 % and v is assigned as the tax paremeter

params = {'theta':0.67, 'delta':0.0625, 'h':0.31, 'beta':0.9804, 'z':1.36, 'v':0.04}
ramsey = ramseyModel(params)
solution=ramsey.get_numericSteadyState(k0=4, c0=1)
k_star=solution[0]
c_star=solution[1]
y_star=0.0625*k_star+c_star
v=k_star/y_star
i_star=0.0625*k_star/y_star



# Values at steady state 
print('Capital Old Steady State after tax = '"%.2f" % (solution[0])) 
print('Consumption  Old Steady State after tax = '"%.2f" % (solution[1])) 
print('Output Old Steady State after tax = '"%.2f" % y_star) 
print('Capital per output Old Steady State after tax = '"%.2f" % v) 
print('Investment per output  Old Steady State after tax = '"%.2f" % i_star)

## Part E2 : introduction of capital tax in new steady state
params = {'theta':0.67, 'delta':0.0625, 'h':0.31, 'beta':0.9804, 'z':3.26, 'v':0.04}
ramsey = ramseyModel(params)
solution=ramsey.get_numericSteadyState(k0=4, c0=1)
k_new=solution[0]
c_new=solution[1]
y_new=0.0625*k_new+c_new
v=k_new/y_new
i_new=0.0625*k_new/y_new


# Values at steady state 
print('Capital New Steady State after tax = '"%.2f" % (solution[0])) 
print('Consumption New Steady State after tax = '"%.2f" % (solution[1])) 
print('Output New Steady State after tax = '"%.2f" % y_new) 
print('Capital per output New Steady State after tax = '"%.2f" % v) 
print('Investment per output  New Steady State after tax = '"%.2f" % i_new) 


# Transition

#Compute the transition
v=0.04
delta=0.0625
z2=3.26
theta=0.67
beta=0.9804
time=10
kt = np.zeros(time)
yt = np.zeros(time)
ct = np.zeros(time)
it = np.zeros(time)

kt[0] = 1.97
ct[0] = 0.57


for t in range(time - 1):
    kt[t+1] = pow(kt[t], 1-theta)*pow(z2*h, theta) - ct[t] + (1-delta-v)*kt[t]
    ct[t+1] = ct[t]*beta*(1 - delta - v + (1-theta)*pow(z2*h, theta)*pow(kt[t], -theta))
   
kt=kt[t+1]
ct=ct[t+1]
print('Transition of capital with permanent capital tax = '"%.2f" % kt)
print('Transition of consumption with permanent capital tax = '"%.2f" % ct)
    


### Exercise 2.1: Closed economy

# For all our calculations we assume k=1 for both countries.

#Import packages
try:
    from pip import main as pipmain
except:
    from pip._internal import main as pipmain
pipmain(['install','gekko'])
from gekko import GEKKO

# Country A 
m = GEKKO()
kappa=m.Param(value=5.0)
v=m.Param(value=1)
sigma=m.Param(value=0.8)
eta_l =m.Param(value=5.5)
eta_h =m.Param(value=0.5)
z =m.Param(value=1)
theta =m.Param(value=0.6)
kmax =m.Param(value=2)
lambd =m.Param(value=0.95)
phi = m.Param(value=0.2)
k=m.Param(value=1)
c_l,c_h,h_l,h_h,w,r=[m.Var(1) for i in range(6)]
m.Equations ([-r+(1-theta)*z*(h_l*eta_l+h_h*eta_h)**(theta)*kmax**(-theta)==0,\
              -w+(theta)*z*kmax**(1-theta)*(h_l*eta_l+h_h*eta_h)**(theta-1)==0,\
              -kappa*h_l**(1/v)+c_l**(-sigma)*w*eta_l*lambd*(1-phi)*(w*h_l)**(-phi)==0,\
              -kappa*h_h**(1/v)+c_h**(-sigma)*w*eta_h*lambd*(1-phi)*(w*h_h)**(-phi)==0,\
              -c_l+lambd*(w*h_l*eta_l)**(1-phi)+r*k**(eta_l)==0,\
              -c_h+lambd*(w*h_h*eta_h)**(1-phi)+r*k**(eta_h)==0])
        
m.solve(disp=False)
print(c_l.value,c_h.value,h_l.value,h_h.value,w.value,r.value)

# Country B
m = GEKKO()
kappa=m.Param(value=5.0)
v=m.Param(value=1)
sigma=m.Param(value=0.8)
eta_l =m.Param(value=3.5)
eta_h =m.Param(value=2.5)
z =m.Param(value=1)
theta =m.Param(value=0.6)
kmax =m.Param(value=2)
lambd =m.Param(value=0.84)
phi = m.Param(value=0.2)
k=m.Param(value=1)
c_l,c_h,h_l,h_h,w,r=[m.Var(1) for i in range(6)]
m.Equations ([-r+(1-theta)*z*(h_l*eta_l+h_h*eta_h)**(theta)*kmax**(-theta)==0,\
              -w+(theta)*z*kmax**(1-theta)*(h_l*eta_l+h_h*eta_h)**(theta-1)==0,\
              -kappa*h_l**(1/v)+c_l**(-sigma)*w*eta_l*lambd*(1-phi)*(w*h_l)**(-phi)==0,\
              -kappa*h_h**(1/v)+c_h**(-sigma)*w*eta_h*lambd*(1-phi)*(w*h_h)**(-phi)==0,\
              -c_l+lambd*(w*h_l*eta_l)**(1-phi)+r*k**(eta_l)==0,\
              -c_h+lambd*(w*h_h*eta_h)**(1-phi)+r*k**(eta_h)==0])
        
m.solve(disp=False)
print(c_l.value,c_h.value,h_l.value,h_h.value,w.value,r.value)

## Ex 2.2: Adding labor unions
m = GEKKO()
kappa=m.Param(value=5.0)
v=m.Param(value=1)
sigma=m.Param(value=0.8)
eta_la =m.Param(value=5.5)
eta_ha =m.Param(value=0.5)
eta_lb =m.Param(value=3.5)
eta_hb =m.Param(value=2.5)
z =m.Param(value=1)
theta =m.Param(value=0.6)
lambd_a =m.Param(value=0.95)
lambd_b =m.Param(value=0.84)
phi = m.Param(value=0.2)
k_la=m.Param(value=1)
k_ha=m.Param(value=1)
k_lb=m.Param(value=1)
k_hb=m.Param(value=1)
c_la,c_ha,h_la,h_ha,w_a,r_a,c_lb,c_hb,h_lb,h_hb,w_b,r_b,k_lsa,k_lsb,k_hsa,k_hsb=[m.Var(1) for i in range(16)]

m.Equations ([-kappa*h_la**(1/v)+c_la**(-sigma)*w_a*eta_la*lambd_a*(1-phi)*(w_a*h_la)**(-phi)==0,\
               -kappa*h_ha**(1/v)+c_ha**(-sigma)*w_a*eta_ha*lambd_a*(1-phi)*(w_a*h_ha)**(-phi)==0,\
               -r_a+(1-theta)*z*(k_lsa +k_hsa +(k_lb-k_lsb)+(k_hb-k_hsb))**(-theta)*(h_la*eta_la+h_ha*eta_ha)**(theta)==0,\
               -w_a+(theta)*z*(k_lsa +k_hsa +(k_lb-k_lsb)+(k_hb-k_hsb))**(1-theta)*(h_la*eta_la+h_ha*eta_ha)**(theta-1)==0,\
               -r_b+r_a*eta_ha*(k_hsa)**(eta_ha-1)==0,\
               -c_la+lambd_a*(w_a*h_la*eta_la)**(1-phi)+r_a*(k_lsa**(eta_la))+r_b*(k_la-k_lsa)==0, \
               -c_ha+lambd_a*(w_a*h_ha*eta_ha)**(1-phi)+r_a*(k_hsa**(eta_ha))+r_b*(k_ha-k_hsa)==0,\
               -r_b+r_a*eta_la*(k_lsa)**(eta_la-1)==0,\
               -kappa*h_lb**(1/v)+c_lb**(-sigma)*w_b*eta_lb*lambd_b*(1-phi)*(w_b*h_lb)**(-phi)==0, \
               -kappa*h_hb**(1/v)+c_hb**(-sigma)*w_b*eta_hb*lambd_b*(1-phi)*(w_b*h_hb)**(-phi)==0,\
               -r_b+(1-theta)*z*(k_lsb +k_hsb +(k_la-k_lsa)+(k_ha-k_hsa))**(-theta)*(h_lb*eta_lb+h_hb*eta_hb)**(theta)==0,\
               -w_b+(theta)*z*(k_lsb +k_hsb +(k_la-k_lsa)+(k_ha-k_hsa))**(1-theta)*(h_lb*eta_lb+h_hb*eta_hb)**(theta-1)==0,\
               -r_a+r_b*eta_lb*(k_lsb)**(eta_lb-1)==0,\
               -r_a+r_a*eta_hb*(k_hsb)**(eta_hb-1)==0,\
               -c_lb+lambd_b*(w_b*h_lb*eta_lb)**(1-phi)+r_b*(k_lsb**(eta_lb))+r_a*(k_lb-k_lsb)==0,\
               -c_hb+lambd_b*(w_b*h_hb*eta_hb)**(1-phi)+r_b*(k_hsb**(eta_hb))+r_a*(k_hb-k_hsb)==0])

m.solve(disp=False)
print(c_la.value,c_ha.value,h_la.value,h_ha.value,w_a.value,r_a.value,c_lb.value,c_hb.value,h_lb.value,h_hb.value,w_b.value,r_b.value,k_lsa.value,k_lsb.value,k_hsa.value,k_hsb.value) 


