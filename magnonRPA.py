### Evaluates RPA propagators for magnons due to interaction with holes
### Jonathan Curtis 
### 03/22/25

import numpy as np
from scipy import integrate as intg
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr

from collections import namedtuple

params = namedtuple('params','t mu T NMC Emax')

### Plotting settings 
#plt.rc('figure', dpi=100)
#plt.rc('figure',figsize=(4,1.7))
plt.rc('font', family = 'Times New Roman')
plt.rc('font', size = 14)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=18)
plt.rc('lines', linewidth=2.5)

figDirectory = "../figures/"
dataDirectory = "../data/"

S = 0.5
zero = 1.e-8

rng = np.random.default_rng()

### Hole spectral function for particular energy and momentum 
### This should be ideally a callable function which is used to sample energy and momenta and is specified by the model
### Here k = [E,kx,ky]
def hole_spectrum(k,ps):
	E=k[0]
	t = ps.t
	mu = ps.mu

	W = 8.*t ### square lattice bandwidth in terms of hopping 
	### This model uses a purely incoherent, flat band with box density of states, just for testing 

	if E < -W/2. - mu/2.:
		return 0.

	elif E > W/2. -mu/2.:
		return 0.

	else: 
		return 1./W 

### This method will calculate the density of holes for a particular chemical potential, temperature, and spectral function 
def calc_density(ps):
	ks = hole_sample(ps) 

	occs = 0.5-0.5*np.tanh(ks[0,:]/(2.*ps.T))

	return np.mean(occs)

### This method performs an MC sampling from the hole spectral function 
### It returns N samples 
### The return is in the form of [E,kx,ky] x N (3,N) shape 
def hole_sample(ps):
	N = int(ps.NMC)
	samples = np.zeros((3,N))

	num_accept = 0
	while num_accept < N:
		k = rng.random(2)*2.*np.pi-np.pi ### Random k point to propose
		e = rng.random()*2.*ps.Emax- ps.Emax

		r = rng.random() ### accept probability

		if r < hole_spectrum(np.array([ e,k[0],k[1] ]),ps):
			samples[:,num_accept] = e
			samples[1:,num_accept] = k
			num_accept += 1

	return samples

### Ag form factor 
def Ag(k):
	return 0.5*np.cos(k[0]) + 0.5*np.cos(k[1])

### B2g form factor 
def B2g(k):
	return 0.5*np.cos(k[0]) - 0.5*np.cos(k[1])

### B1g form factor 
def B1g(k):
	return np.sin(k[0])*np.sin(k[1])

### This method uniformly samples the momentum points 
def gen_kpts_uniform(ps):
	N = int(ps.NMC)
	print(N)
	kpts = rng.random((2,N))*2.*np.pi-np.pi*np.ones((2,N)) ### Random k points 

	return kpts

### This method samples momentum points according to the B2g form factor 
def gen_kpts_B2g(ps):
	N = int(ps.NMC)
	### 
	kpts = np.zeros((2,N))
	num_accept = 0

	while num_accept < N:
		k = rng.random(2)*2.*np.pi-np.pi ### Random k point to propose

		r = rng.random() ### accept probability
		#r = 0.

		if r < (B2g(k))**2:
			kpts[:,num_accept] = k
			num_accept += 1

	return kpts

### This function evalutes the imaginary part of the RPA response function for a given magnon frequency and momentum  
### These are passed as q = [w,qx,qy]
def ImPi(q,ps):

	### We need to evaluate the integrals 
	### integral_{E,p} iA(E,p) A(E+w,p+q) gamma_{p} [ gamma_{p}, gamma_{p+q} ] ( f(E)-f(E') )/(E - E'+w)

	### We start by Monte Carlo sampling the hole momentum and energy k,E according to the spectral function 

	N = int(ps.NMC)
	k1s = hole_sample(ps) ### This generates a sample of energies and momenta for holes 

	k2s = k1s + np.tensordot(q,np.ones(N),axes=-1) ### This is the sum of the energy and momenta 

	### The only part that differs is the form factor 

	integrands= np.zeros(N) 
	for i in range(N):
		integrands[i] = hole_spectrum(k2s[:,i],ps)*( 0.5*np.tanh(0.5*k1s[0,i]/ps.T) - 0.5*np.tanh(0.5*k2s[0,i]/ps.T) )

	return 2.*np.pi*S*ps.t**2*np.array([ np.mean(integrands*(Ag(k1s[1:,:]))**2 ), np.mean(integrands*Ag(k1s[1:,:])*Ag(k2s[1:,:]) )  ]) 


def main():


	t = 1.
	mu = -7.3*t ### Approximately 4% doping
	T = 0.11*t
	U = 7.5*t
	J = 4.*t**2/U
	NMC = 1e4
	Emax = 10.*t

	p = params(t,mu,T,NMC,Emax)

	delta = calc_density(p)
	print(delta)

	numws = 30
	ws = np.linspace(-4.*J,4.*J,numws)
	ImPis = np.zeros((2,numws))
	
	for i in range(numws):
		q = np.array([ws[i],np.pi,0.])
		ImPis[:,i] = ImPi(q,p)


	
	plt.plot(ws/J,ImPis[0,:],label=r'Im$\Pi_0$')
	plt.plot(ws/J,ImPis[1,:],label=r'Im$\Pi_1$')
	#plt.yscale('log')
	plt.xlabel(r'$\omega/J$')
	plt.legend()
	plt.show()



if __name__ == "__main__":
	main()





