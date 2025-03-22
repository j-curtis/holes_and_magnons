### Squeezed ansatz for spin waves in nematic quench of Heisenberg model 
### Jonathan Curtis 
### 03/05/25

import numpy as np
from scipy import integrate as intg
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr


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

rng = np.random.default_rng()

### d-wave form factor 
def dff(k):
	return 0.5*np.cos(k[0]) - 0.5*np.cos(k[1])

### B1g form factor 
def B1gff(k):
	return np.sin(k[0])*np.sin(k[1])


### s-wave form factor 
def sff(k):
	return 0.5*np.cos(k[0]) + 0.5*np.cos(k[1])

### This method samples the momentum points according to the d-wave form factor 
def gen_kpts(N):
	### 
	kpts = np.zeros((2,N))
	num_accept = 0

	while num_accept < N:
		k = rng.random(2)*2.*np.pi-np.pi ### Random k point to propose

		r = rng.random() ### accept probability
		#r = 0.

		if r < (dff(k))**2:
		#if r < (B1gff(k))**2:
			kpts[:,num_accept] = k
			num_accept += 1

	return kpts

### This computes the energy given the momentum and J
def energy(k,J):
	
	return 2.*S*J*np.sqrt(1.-sff(k)**2 ) 


def main():
	t = 1.
	U = 7.5
	J = 4.*t**2/U

	N = 10000 ### number of momentum points we sample 

	kpts = gen_kpts(N)

	energies = 2.*energy(kpts,J) ### 2 because there is degeneracy between k and - k pair 

	nts = 400
	times = np.linspace(0.,50.,nts)
	time_domain_signal = np.zeros(nts,dtype=complex)

	for i in range(N):
		time_domain_signal[:] += np.exp(-1.j*energies[i]*times)/float(N)


	plt.scatter(kpts[0,:],kpts[1,:],s=.2)
	plt.xlabel(r'$k_x$')
	plt.ylabel(r'$k_y$')
	plt.xticks([-np.pi,0.,np.pi],[r'$-\pi$',r'$0$',r'$\pi$'])
	plt.yticks([-np.pi,0.,np.pi],[r'$-\pi$',r'$0$',r'$\pi$'])
	#plt.savefig(figDirectory+"sampled_momenta.pdf",bbox_inches='tight')
	plt.show()

	plt.hist(energies/J,bins=100,density=False)
	plt.xlabel(r'$E/J$')
	plt.xlim(1.7,2.05)
	plt.ylabel(r'$N(E)$ [arb. units]')
	#plt.savefig(figDirectory+"nematicity_spectrum.pdf",bbox_inches='tight')
	plt.show()


	plt.plot(times*J,np.real(time_domain_signal))
	plt.xlabel(r'$tJ$')
	plt.xlim(0.,50.*J)
	plt.ylabel(r'$S(t)$ [arb. units]')
	#plt.savefig(figDirectory+"bimagnon_time_domain.pdf",bbox_inches='tight')
	plt.show()

	

if __name__ == "__main__":
	main()





