import math
import numpy as np
import scipy as s
import scipy.integrate as q
import matplotlib.pyplot as plt
#Constants
H0 = 2.19507453e-18 #using 67.74
Wm = 0.279
Wq = 1 - Wm
w = -1

#Basic Functions
def H(a):
    return H0*np.sqrt(Wm*a**(-3) + Wq*a**(-3*(1+w)))

def dH(a):
    return (H0**2/(2*H(a)))*(-3*Wm*a**(-4) - 3*(1 + w)*Wq*a**(-3*w -4))

def Dplus(a):
    def integrand(x):
        return (x*H(x))**(-3)
    [y,err] = q.quad(integrand,0,a)
    return 2.5*H0**2*Wm*H(a)*y

def fplus(a):
    h = H(a)
    dh = dH(a)
    d = Dplus(a)
    return (a/d)*((dh*d/h) + 2.5*H0**2*Wm/(h**2*a**3))

def fminus(a):
    return dH(a)*a/H(a)

#Primary Functions
#Code to call values of n = 2
def z2(y,a):
    fp = fplus(a)
    fm = fminus(a)
    return [fp/a*(2 + y[1] - 2*y[0]) , fp/a*((2/3) + (fm*fp**(-2))*(y[1] - y[0]) - y[1])]

def y2(x):
    a = np.linspace(a0,x,N)
    sol = q.odeint(z2,y20,a)
    return [sol[-1,0], sol[-1,1]]

#Let y = [nu,mu], z = dy/dt
#This is code for n = 3
def z3(y,a):
    [nu2,mu2] = y2(a)
    fp = fplus(a)
    fm = fminus(a)
    return [fp/a*(y[1] - 3*y[0] + 3*(nu2 + mu2)), fp/a*(-2*y[1] + (fm*fp**(-2))*(y[1] - y[0]) + 2*mu2)]

#Main Code
#Initial conditions - Using EdS values
y20 = [34/21,26/21]
y30 = [682/189, 142/63]

#Integration
a0 = 0.01
N = 10000
a = np.linspace(a0,1,N)
soln = q.odeint(z3,y30,a)

nuads = []
muads = []
for i in range(0,N):
    nuads.append(y30[0])
    muads.append(y30[1])

#Plotting
plt.figure(1)
plt.subplot(111)
plt.plot((1-a)/a,soln[:,0], 'k', label = r'$\nu_3$')
plt.plot((1-a)/a, nuads, '--r', label = r'$\nu_{EdS}$')
plt.title(r'$\nu_3$ as a function of redshift', fontsize=20)
plt.xlabel(r'$z$', fontsize=20)
plt.ylabel(r'$\nu_3$', fontsize=20)
plt.legend(fontsize=15)
plt.xlim([0,3])

plt.figure(2)
plt.subplot(111)
plt.plot((1-a)/a,soln[:,1], 'k', label=r'$\mu_3$')
plt.plot((1-a)/a, muads, '--r', label=r'$\mu_{EdS}$')
plt.title(r'$\mu_3$ as a function of redshift', fontsize=20)
plt.xlabel(r'$z$', fontsize=20)
plt.ylabel(r'$\mu_3$', fontsize=20)
plt.legend(fontsize=15)
plt.xlim([0,3])

plt.show()
