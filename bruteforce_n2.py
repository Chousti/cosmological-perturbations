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
#Let y = [nu,mu], z = dy/dt
#This is code for n = 2
def z(y,a):
    fp = fplus(a)
    fm = fminus(a)
    return [fp/a*(2 + y[1] - 2*y[0]) , fp/a*((2/3) + (fm*fp**(-2))*(y[1] - y[0]) - y[1])]

#Main Code
#Initial conditions - Using EdS values
y0 = [34/21,26/21]

#Integration
a = np.linspace(0.01,1,100000)
sol = q.odeint(z,y0,a)
nuads = []
muads = []
for i in range(0,100000):
    nuads.append(y0[0])
    muads.append(y0[1])

#Plotting
plt.figure(1)
plt.subplot(111)
plt.plot((1-a)/a,sol[:,0], 'k', label = r'$\nu_2$')
plt.plot((1-a)/a, nuads, '--r', label = r'$\nu_{EdS}$')
plt.title(r'$\nu_2$ as a function of redshift', fontsize=20)
plt.xlabel(r'$z$', fontsize=20)
plt.ylabel(r'$\nu_2$', fontsize=20)
plt.legend(fontsize=15)
plt.xlim([0,3])

plt.figure(2)
plt.subplot(111)
plt.plot((1-a)/a,sol[:,1], 'k', label=r'$\mu_2$')
plt.plot((1-a)/a, muads, '--r', label=r'$\mu_{EdS}$')
plt.title(r'$\mu_2$ as a function of redshift', fontsize=20)
plt.xlabel(r'$z$', fontsize=20)
plt.ylabel(r'$\mu_2$', fontsize=20)
plt.legend(fontsize=15)
plt.xlim([0,3])

plt.show()
