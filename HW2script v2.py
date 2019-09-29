import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from numpy import inf
plt.style.use("ggplot")

### Ex.1 Taylor approximations

## Ex1.1
x = sy.Symbol('x')
def f(x):
	return x**0.321

# Taylor function
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)
def taylor(function, x0, n, x = sy.Symbol('x')):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x, i).subs(x, x0))/(factorial(i))*(x - x0)**i
        i += 1
    return p
# Ploting for diffrent degrees
def plot():
    x_lims = [0,4]
    x1 = np.linspace(x_lims[0],x_lims[1],800)
    y1 = []
    degree = [1, 2, 5, 20]
    for j in degree:
        func = taylor(f(x),1,j)
        print('Taylor expansion at n='+str(j),func)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='order '+str(j))
        y1 = []
    plt.plot(x1,f(x1),label='f(x)')
    plt.xlim(x_lims)
    plt.ylim([0,4])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('TS approximation')
    plt.show()
plot()
plt.clf()

## Ex1.2
x = sy.Symbol('x', real=True)
def g(x):
	return 0.5*(x + abs(x))

# Ploting for diffrent degrees
plt.figure(2)
def plot():
    x_lims = [-2,6]
    x1 = np.linspace(x_lims[0],x_lims[1],800)
    y1 = []
    degree = [1, 2, 5, 20]
    for j in degree:
        func = taylor(g(x),2,j)
        print('Taylor expansion at n='+str(j),func)
        for k in x1:
            y1.append(func.subs(x,k))
        plt.plot(x1,y1,label='order '+str(j))
        y1 = []
    plt.plot(x1,g(x1),label='g(x)')
    plt.xlim(x_lims)
    plt.ylim([-2,6])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('TS approximation Ramp function')
    plt.show()
plot()   



## Ex1.3a
# Runge function
def h(x):
	return 1 / (1 + 25 * x**2)

# evenly spaced interpolation nodes
x_lims=[-1,1]
x1 = np.linspace(x_lims[0],x_lims[1],11) 
yh= h(x1)
xs = np.linspace(x_lims[0],x_lims[1],101)
cs = poly.polyfit(x1,yh,3)
ffit3 = poly.polyval(xs,cs)

# Monomials of order 5 and 10:
coeffs5 = poly.polyfit(x1,yh,5)
ffit5 = poly.polyval(xs,coeffs5)
coeffs10 = poly.polyfit(x1,yh,10)
ffit10 = poly.polyval(xs,coeffs10)

# exponential function
def i(x):
	with np.errstate(divide='ignore', invalid='ignore'):
		return np.exp(1 / x)
yi = i(x1)
yi[yi == inf] = 9999
csi = poly.polyfit(x1, yi,3)
coeffi = poly.polyfit(x1, yi, 5)
ffiti = poly.polyval(xs,csi)
ffit3i = poly.polyval(xs, coeffi)

# Monomials
coeffs10i = poly.polyfit(x1,yi,10)
ffit10i = poly.polyval(xs,coeffs10i)

# ramp function
def j(x):
	return .5*(x+abs(x))
yj = j(x1)

csj = poly.polyfit(x1,yj,3)
ffit3j = poly.polyval(xs,csj)

# Monomials of order 5 and 10:
coeffs5j = poly.polyfit(x1,yj,5)
ffit5j = poly.polyval(xs,coeffs5j)
coeffs10j = poly.polyfit(x1,yj,10)
ffit10j = poly.polyval(xs,coeffs10j)

# Plots
fig = plt.figure(figsize=(10,10))
fig.suptitle('Function Approximation: Evenly-Spaced nodes & monomials')

ax1 = plt.subplot(321)
plt.plot(x1,yh,'o',label='data')
plt.plot(xs,h(xs), label='True')
plt.plot(xs,ffit3,label='Cubic Polynomials') 
plt.plot(xs,ffit5,label='Monomial5')
plt.plot(xs,ffit10,label='Monomial10')
plt.legend(bbox_to_anchor=(0.5,0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)
plt.ylim(-0.2,1)
plt.title('Runge Function')

plt.subplot(322, sharex=ax1)
plt.plot(xs,h(xs)-ffit3,label='Cubic error') 
plt.plot(xs,h(xs)-ffit5,label='m5 error')
plt.plot(xs,h(xs)-ffit10,label='m10 error')
plt.ylim(0,1)
plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)
plt.title('Runge Function Approximation Errors')

plt.subplot(323, sharex=ax1)
plt.plot(x1,yi,'o')
plt.plot(xs,i(xs))
plt.plot(xs,ffit3i) 
plt.plot(xs,ffiti)
plt.plot(xs,ffit10i)
plt.ylim(-100,1000)
plt.title('Exponential Function ')

plt.subplot(324, sharex=ax1)
plt.plot(xs,i(xs)-ffit3i) 
plt.plot(xs,i(xs)-ffiti)
plt.plot(xs,i(xs)-ffit10i)
plt.ylim(0,100)
plt.title('Exponential Function Approximation Errors')


plt.subplot(325, sharex=ax1)
plt.plot(x1,yj,'o')
plt.plot(xs,j(xs))
plt.plot(xs,ffit3j) 
plt.plot(xs,ffit5j)
plt.plot(xs,ffit10j)
plt.ylim(-0.5,2.5)
plt.title('Ramp Function')

plt.subplot(326, sharex=ax1)
plt.plot(xs,j(xs)-ffit3j) 
plt.plot(xs,j(xs)-ffit5j)
plt.plot(xs,j(xs)-ffit10j)
plt.ylim(0,0.25)
plt.title('Ramp Function Approximation Errors')
plt.show()

# Ex 1.3b

import numpy as np
import matplotlib.pyplot as plt

# Exponential function
def f(x):
    f = 0.5*(x + abs(x))
    return f

domain = np.linspace(-1, 1, 100)


# Chebshev nodes with 20
n = 20
i = np.arange(n, dtype=np.float64)
xn= []

for i in range(n):
   x = np.cos((2 *(i+ 1) - 1) / (2 * n) * np.pi)
   xn.append(x)
  
x= np.asarray(xn)

# Cubic polynomial
p_cubic=np.polyfit(x,f(x),3)
y_cubic=np.polyval(p_cubic,domain) 

# Monomial of order 5
p_order5=np.polyfit(x,f(x),5)
y_order5=np.polyval(p_order5,domain)

#3 Monomial of order 10
p_order10=np.polyfit(x,f(x),10)
y_order10=np.polyval(p_order10,domain)

#Calculation of approximation errors 
error_cubic=abs(f(domain)-y_cubic)
error_order5=abs(f(domain)-y_order5)
error_order10=abs(f(domain)-y_order10)

# Plot 
plt.figure(4)

plt.subplot(421)
plt.plot(domain, f(domain), color='pink', label='ramp function')
plt.plot(domain, y_cubic,'r', label='cubic')
plt.plot(domain, y_order5, color='orange' , label='monomial 5')
plt.plot(domain, y_order10, color='blue', label='monomial 10')
plt.legend(['ramp', 'cubic polynomial', 'monomial 5', 'monomial 10'], loc='upper right')
plt.title('Ramp function Chebychev nodes')

plt.subplot(422)
plt.plot(domain, error_cubic,'r', label='cubic')
plt.plot(domain, error_order5, color='orange', label='monomial 5')
plt.plot(domain, error_order10, color='blue', label='monomial 10')
plt.legend(loc='upper right')
plt.title('Ramp function errors Chebychev nodes')

plt.subplots_adjust(top=4.7, bottom=0.1, left=0, right=1.8, hspace=0.7, wspace=0.7)


# Ramp function
def f(x):
    f = 1/(1+25*(x**2))
    return f

domain = np.linspace(-1, 1, 100)

# Chebychev interpolation nodes
n = 20
i = np.arange(n, dtype=np.float64)
xn= []

for i in range(n):
   x = np.cos((2 *(i+ 1) - 1) / (2 * n) * np.pi) 
   xn.append(x)
  
x= np.asarray(xn)

# Cubic polynomial
p_cubic=np.polyfit(x,f(x),3) 
y_cubic=np.polyval(p_cubic,domain) 

# Monomial of order 5
p_order5=np.polyfit(x,f(x),5)
y_order5=np.polyval(p_order5,domain)

# Monomial of order 10
p_order10=np.polyfit(x,f(x),10)
y_order10=np.polyval(p_order10,domain)

# Calculation of approximation errors
error_cubic=abs(f(domain)-y_cubic)
error_order5=abs(f(domain)-y_order5)
error_order10=abs(f(domain)-y_order10)

# Plot 
plt.figure(5)

plt.subplot(521)
plt.plot(domain, f(domain), color='pink', label='runge function')
plt.plot(domain, y_cubic,'r', label='cubic')
plt.plot(domain, y_order5, color='orange' , label='monomial 5')
plt.plot(domain, y_order10, color='blue', label='monomial 10')
plt.legend(['runge', 'cubic polynomial', 'monomial 5', 'monomial 10'], loc='upper right')
plt.title('Runge function Chebychev nodes')

plt.subplot(522)
plt.plot(domain, error_cubic,'r', label='cubic')
plt.plot(domain, error_order5, color='orange', label='monomial 5')
plt.plot(domain, error_order10, color='blue', label='monomial 10')
plt.legend(loc='upper right')
plt.title('Runge function errors Chebychev nodes')

plt.subplots_adjust(top=6.0, bottom=0.1, left=0, right=1.8, hspace=0.7, wspace=0.7)

#Definition of the function we want to approximate 
def f(x):
    f = np.exp(1/x)
    return f
domain = np.linspace(-1, 1, 100)

# Chebychev interpolation nodes
n = 20
i = np.arange(n, dtype=np.float64)
xn= []

for i in range(n):
   x = np.cos((2 *(i+ 1) - 1) / (2 * n) * np.pi)
   xn.append(x)
  
x= np.asarray(xn)

# Cubic polynomial
p_cubic=np.polyfit(x,f(x),3) 
y_cubic=np.polyval(p_cubic,domain) 

#2 Monomial of order 5
p_order5=np.polyfit(x,f(x),5)
y_order5=np.polyval(p_order5,domain)

#3 CASE: Monomial of order 10
p_order10=np.polyfit(x,f(x),10)
y_order10=np.polyval(p_order10,domain)

#Calculation of approximation errors 
error_cubic=abs(f(domain)-y_cubic)
error_order5=abs(f(domain)-y_order5)
error_order10=abs(f(domain)-y_order10)

# Plot 
plt.figure(6)

plt.subplot(621)
plt.plot(domain, f(domain), color='pink', label='exponential function')
plt.plot(domain, y_cubic,'r', label='cubic')
plt.plot(domain, y_order5, color='orange' , label='monomial 5')
plt.plot(domain, y_order10, color='blue', label='monomial 10')
plt.ylim([-20000,200000])
plt.legend(['exponential f', 'cubic polynomial', 'monomial 5', 'monomial 10'], loc='upper right')
plt.title('Exponential function approximation Chebychev nodes')

plt.subplot(622)
plt.plot(domain, error_cubic,'r', label='cubic')
plt.plot(domain, error_order5, color='orange', label='monomial 5')
plt.plot(domain, error_order10, color='blue', label='monomial 10')
plt.legend(loc='upper right')
plt.ylim([0,200000])
plt.title('Exponential function errors Chebychev nodes')

plt.subplots_adjust(top=6.0, bottom=0.1, left=0, right=1.8, hspace=0.7, wspace=0.7)

## Ex 1.3c

# Exponential function
def f(x):
    f = 0.5*(x + abs(x))
    return f

domain = np.linspace(-1, 1, 100)

#  Chebychev interpolation nodes
n = 20 #arbitrarly chosen
i = np.arange(n, dtype=np.float64)
xnode= []

for i in range(n):
   x = np.cos((2 *(i+ 1) - 1) / (2 * n) * np.pi) 
   xnode.append(x)
  
x= np.asarray(xnode)

# Interpolation with Chebychev polynomials

# Order 3
p_cubic_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),3)
y_cubic_chebychev=np.polynomial.chebyshev.chebval(domain,p_cubic_chebychev) 

# Order 5
p_ord5_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),5)
y_ord5_chebychev=np.polynomial.chebyshev.chebval(domain,p_ord5_chebychev) 

# Order 10
p_ord10_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),10)
y_ord10_chebychev=np.polynomial.chebyshev.chebval(domain,p_ord10_chebychev)

# Calculation of approximation errors
error_cubic=abs(f(domain)-y_cubic_chebychev)
error_ord5=abs(f(domain)-y_ord5_chebychev)
error_ord10=abs(f(domain)-y_ord10_chebychev)

#Plot
plt.figure(7)

plt.subplot(721)
plt.plot(domain, f(domain), color='pink', label='ramp function')
plt.plot(domain, y_cubic_chebychev,'r', label='cubic')
plt.plot(domain, y_ord5_chebychev, color='orange' , label='order 5')
plt.plot(domain, y_ord10_chebychev, color='blue', label='order 10')
plt.legend(['ramp', 'order 3', 'order 5', 'order 10'], loc='upper right')
plt.title('Ramp function Chebychev nodes & polynomial')

plt.subplot(722)
plt.plot(domain, error_cubic,'r', label='order 3')
plt.plot(domain, error_ord5, color='orange', label='order 5')
plt.plot(domain, error_ord10, color='blue', label='order 10')
plt.legend(loc='upper right')
plt.title('Ramp function errors Chebychev nodes & polynomial')

plt.subplots_adjust(top=6.0, bottom=0.1, left=0, right=1.8, hspace=0.7, wspace=0.7)

# Runge function
def f(x):
    f = 1/(1+25*(x**2))
    return f
domain = np.linspace(-1, 1, 100)


# Chebychev interpolation nodes
n = 20 
i = np.arange(n, dtype=np.float64)
xnode= []

for i in range(n):
   x = np.cos((2 *(i+ 1) - 1) / (2 * n) * np.pi) 
   xnode.append(x)
  
x= np.asarray(xnode)

# Interpolation with Chebychev polynomials

# Order 3
p_cubic_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),3)
y_cubic_chebychev=np.polynomial.chebyshev.chebval(domain,p_cubic_chebychev) 

# Order 5
p_ord5_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),5)
y_ord5_chebychev=np.polynomial.chebyshev.chebval(domain,p_ord5_chebychev) 

#3 Order 10
p_ord10_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),10)
y_ord10_chebychev=np.polynomial.chebyshev.chebval(domain,p_ord10_chebychev)

# Calculation of approximation errors 
error_cubic=abs(f(domain)-y_cubic_chebychev)
error_ord5=abs(f(domain)-y_ord5_chebychev)
error_ord10=abs(f(domain)-y_ord10_chebychev)

#Plot 
plt.figure(8)

plt.subplot(821)
plt.plot(domain, f(domain), color='pink', label='runge function')
plt.plot(domain, y_cubic_chebychev,'r', label='cubic')
plt.plot(domain, y_ord5_chebychev, color='orange' , label='order 5')
plt.plot(domain, y_ord10_chebychev, color='blue', label='order 10')
plt.legend(['runge', 'order 3', 'order 5', 'order 10'], loc='upper right')
plt.title('Runge function Chebychev nodes  polynomial')

plt.subplot(822)
plt.plot(domain, error_cubic,'r', label='order 3')
plt.plot(domain, error_ord5, color='orange', label='order 5')
plt.plot(domain, error_ord10, color='blue', label='order 10')
plt.legend(loc='best')
plt.title('Runge function errors Chebychev nodes and polynomial')

plt.subplots_adjust(top=6.0, bottom=0.1, left=0, right=1.8, hspace=0.7, wspace=0.7)

# Ramp function
def f(x):
    f = np.exp(1/x) 
    return f

domain = np.linspace(-1, 1, 100)


# Chebychev interpolation nodes
n = 20 
i = np.arange(n, dtype=np.float64)
xnode= []

for i in range(n):
   x = np.cos((2 *(i+ 1) - 1) / (2 * n) * np.pi) 
   xnode.append(x)
  
x= np.asarray(xnode)

# Interpolation with Chebychev polynomials

# Order 3
p_cubic_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),3)
y_cubic_chebychev=np.polynomial.chebyshev.chebval(domain,p_cubic_chebychev) 

# Order 5
p_ord5_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),5)
y_ord5_chebychev=np.polynomial.chebyshev.chebval(domain,p_ord5_chebychev) 

# Order 10
p_ord10_chebychev=np.polynomial.chebyshev.chebfit(x,f(x),10)
y_ord10_chebychev=np.polynomial.chebyshev.chebval(domain,p_ord10_chebychev)

# Calculation of approximation
error_cubic=abs(f(domain)-y_cubic_chebychev)
error_ord5=abs(f(domain)-y_ord5_chebychev)
error_ord10=abs(f(domain)-y_ord10_chebychev)

#Plot
plt.figure(9)

plt.subplot(921)
plt.plot(domain, f(domain), color='pink', label='exponential f')
plt.plot(domain, y_cubic_chebychev,'r', label='cubic')
plt.plot(domain, y_ord5_chebychev, color='orange' , label='order 5')
plt.plot(domain, y_ord10_chebychev, color='blue', label='order 10')
plt.legend(['exponential f', 'order 3', 'order 5', 'order 10'], loc='upper right')
plt.ylim([-20000,200000])
plt.title('Exponential function Chebychev nodes & polynomial')

plt.subplot(922)
plt.plot(domain, error_cubic,'r', label='order 3')
plt.plot(domain, error_ord5, color='orange', label='order 5')
plt.plot(domain, error_ord10, color='blue', label='order 10')
plt.legend(loc='upper right')
plt.ylim([0,200000])
plt.title('Exponential function errors Chebychev nodes & polynomial')

plt.subplots_adjust(top=8.0, bottom=0.1, left=0, right=1.8, hspace=0.7, wspace=0.7)

