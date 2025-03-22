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

params = namedtuple('params','t mu T k_samples E_samples')


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
def hole_spectrum(E,k,ps):
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
	N = ps.k_samples
	### 
	kpts = rng.random((2,N))*2.*np.pi-np.pi*np.ones((2,N)) ### Random k points 

	return kpts

### This method samples momentum points according to the B2g form factor 
def gen_kpts_B2g(ps):
	N = ps.k_samples
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

### This function evalutes the hole Lindhard function for frequency w and momenta k1,k2 for given parameters 
### This is given by 
### integral dE integral dE' A(E,k1) A(E,k2) ( f(beta E) - f(beta E') )/ (E - E' + w +i0^+)
def lindhard(w,k1,k2,ps):

	### First we Monte Carlo sample energies according to the spectral functions at the particular momenta 

	Emax = 20.*ps.t
	N = ps.E_samples
	es = np.zeros((2,N)) ### This is a set of energies sampled for k1 momenta and k2 momenta 


	num_accept = 0
	while num_accept < N:
		e = rng.random()*Emax - Emax/2.

		r = rng.random() ### accept probability
		#r = 0.

		if r < hole_spectrum(e,k1,ps):
			es[num_accept,0] = e
			num_accept += 1

	num_accept = 0
	while num_accept < N:
		e = rng.random()*Emax - Emax/2.

		r = rng.random() ### accept probability
		#r = 0.

		if r < hole_spectrum(e,k2,ps):
			es[num_accept,1] = e
			num_accept += 1

	### Now we have our energy samples we evaluate the Lindhard function

	lambda f(e) : ( 0.5*np.tanh(e[0]/(2.*ps.T)) - 0.5*np.tanh(e[1]/(2.*ps.T)) )/(e[0] - e[1] + w + 1.j*zero)

	return np.mean(np.vectorize(f(es)))



def main():
	t = 1.
	U = 7.5
	J = 4.*t**2/U

	p= params(t,0.,0.1*t,1e3,1e2)

	kpts = gen_kpts(p)

	energies = 2.*energy(kpts,J) ### 2 because there is degeneracy between k and - k pair 

	plt.scatter(kpts[0,:],kpts[1,:],s=.2)
	plt.xlabel(r'$k_x$')
	plt.ylabel(r'$k_y$')
	plt.xticks([-np.pi,0.,np.pi],[r'$-\pi$',r'$0$',r'$\pi$'])
	plt.yticks([-np.pi,0.,np.pi],[r'$-\pi$',r'$0$',r'$\pi$'])
	plt.show()


if __name__ == "__main__":
	main()





