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
epsilon = 0.0001 ### To regularize the behavior when q = (0,0) or (pi,pi)

### This method generates a momentum space mesh of N x N points
def gen_kpts(N):
	ks = np.linspace(-np.pi,np.pi,N)

	return np.meshgrid(ks,ks)

### Bogoliubov ground state for squeezed ansatz of magnons
### We have in principle squeezed state with eta_q = [etax_q, etay_q] but the ground state has etax_q = 0.
### This is found analytically as a function of the exchanges 
def gs_etas(ks,Jx,Jy):
	### First we generate the appropriate terms 
	kxs = ks[0]
	kys = ks[1]
	N = ks[0].shape[0]

	args = (Jx*np.cos(kxs[:]) + Jy*np.cos(kys[:]) )/(Jx + Jy) 

	### Slightly reduce the results for +1/-1 to regularize the numerics 
	args[args == 1.] = 1. - epsilon
	args[args == -1.] = -1.+epsilon

	etas = np.zeros((2,N,N)) 
	etas[1,:,:] = -np.arctanh(args)

	return etas

### This computes the energy from the etas 
def energy(ks,etas,Jx,Jy):
	kxs = ks[0]
	kys = ks[1]

	etas_mag = np.sqrt( etas[0,:,:]**2 + etas[1,:,:]**2)
	sin_etas_arg = etas[1,:,:]/etas_mag[:,:] 
	us = np.cosh(etas_mag)
	vs = np.sinh(etas_mag)
	integrand =  S*Jx * (us + vs*sin_etas_arg*np.cos(kxs) ) + S*Jy * (us + vs*sin_etas_arg*np.cos(kys) ) - 2.*S*(Jx + Jy)/2. ### Last term comes from a normal ordering correction 

	return np.mean(integrand) ### This should be equivalent to integrating given that the sampling area is 4 pi^2 and the integrals are normalized to this

### This computes the nematicity form the etas of which there are two -- eta = (etax,etay)

### Classical energy of Neel state 
def Neel_energy(Jx,Jy):
	return -S**2*(Jx + Jy)

### This computes the nematicity of the spin waves 
### It returns <S_j . (S_{j+x} - S_{j+y})>
def nematicity(ks,etas):
	kxs = ks[0]
	kys = ks[1]

	etas_mag = np.sqrt( etas[0,:,:]**2 + etas[1,:,:]**2)
	cos_etas_arg = etas[0,:,:]/etas_mag
	sin_etas_arg = etas[1,:,:]/etas_mag
	us = np.cosh(etas_mag)
	vs = np.sinh(etas_mag)

	integrand = S*vs*cos_etas_arg*( np.sin(ks[0]) - np.sin(ks[1]) ) + S*vs*sin_etas_arg*(np.cos(ks[0]) - np.cos(ks[1]))

	return np.mean(integrand)

def main():

	N = 100 ### number of momentum points
	nJs = 20 ### number of J points scanned 


	### Plot on linear scale vs difference 
	deltas = np.linspace(-0.1,0.1,nJs)
	Jxs = np.ones(nJs) + deltas
	Jys = np.ones(nJs) - deltas
	
	energies = np.zeros_like(Jxs)
	nematicities = np.zeros_like(Jxs)

	ks = gen_kpts(N)

	for i in range(nJs):
		etas = gs_etas(ks,Jxs[i],Jys[i])
		energies[i] = energy(ks,etas,Jxs[i],Jys[i]) +Neel_energy(Jxs[i],Jys[i])
		nematicities[i] = nematicity(ks,etas)
		
	np.save(dataDirectory+"Jxs.npy",Jxs)
	np.save(dataDirectory+"Jys.npy",Jys)
	np.save(dataDirectory+"energies.npy",energies)
	np.save(dataDirectory+"nematicities.npy",nematicities)

	plt.plot((Jxs - Jys),energies,color='black',marker='o')
	plt.xlabel(r'$(J_x - J_y)/\frac12(J_x + J_y)$')
	plt.ylabel(r'$\epsilon_{\rm tot}/\frac12(J_x + J_y)$')
	plt.savefig(figDirectory+"energy_total_vs_anisotropy.pdf",bbox_inches='tight')
	plt.show()

	plt.plot((Jxs - Jys),nematicities,color='blue',marker='o')
	plt.xlabel(r'$(J_x - J_y)/\frac12(J_x + J_y)$')
	plt.ylabel(r'$C_{x} - C_{y}$')
	plt.savefig(figDirectory+"nematicity_vs_anisotropy.pdf",bbox_inches='tight')
	plt.show()


if __name__ == "__main__":
	main()





