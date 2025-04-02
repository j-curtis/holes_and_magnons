### Evaluates RPA propagators for magnons due to interaction with holes
### Jonathan Curtis 
### 03/22/25

import numpy as np
import pickle as pkl
import scipy as scp
from scipy import integrate as intg
from scipy import signal 
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


S = 0.5
t = 1. ### We will use units throughout with t = 1 for the time being 
zero = 1.e-5*t ### Infinitesimal broadening factor used to regularize singular magnon kernel. Hopefully should be smaller tha contribution due to holes 

rng = np.random.default_rng()

#######################
### Basic math defs ###
#######################
### Ag form factor (alias for non vector input )
def A1g(kx,ky):
	return 0.5*np.cos(kx) + 0.5*np.cos(ky)

### B2g form factor 
def B2g(kx,ky):
	return 0.5*np.cos(kx) - 0.5*np.cos(ky)

### B1g form factor 
def B1g(kx,ky):
	return np.sin(kx)*np.sin(ky)

### Fermi Dirac function using tanh form 
def fd(w,mu,T):
	return 0.5*np.tanh(0.5*(w-mu)/T)

### Computes spectral function from retarded GF 
def G2A(G):
	return -1./np.pi * np.imag(G)

######################
### Data importing ###
######################

### Loads the numpy pickle files for the hole spectra from Ivan at the specified path location 
### Returns the spectral function arrays kx, ky, w, G as arrays
### The signature of the returned array is rank 3 tensor with indices [kx,ky,w]
def load_hole_spectrum(fpath,roll_momenta=False):
	with open(fpath,'rb') as f:
		kxs,kys,ws,G = pkl.load(f)

	if roll_momenta:
		### we roll the first two axes and the momenta so they are from -pi to pi using np.fft.fftshift

		kxs = kxs - np.pi+kxs[1] -kxs[0]
		kys = kys - np.pi+kys[1] -kys[0]

		G = np.fft.fftshift(G,axes=[0,1])


	return kxs,kys,np.real(ws),np.abs(G2A(G)) ### We need to take abs of spectral function which is directly computed form time-ordered G and needs to be converted to retarded

###################
### Hole Doping ###
###################

### This computes the doping as a function of chemical potential in order to invert and find the chemical potential
def calc_density(kxs,kys,ws,A,mu,T):
	### First we generate the tensor describing the filling factor 
	fd_tensor = gen_fd_tensor(kxs,kys,ws,mu,T)

	### The fd tensor is 0.5 tanh(x/2T)
	### We want 0.5- fd_tensor = 0.5-0.5tanh(x/2) = 1/(e^x + 1)
	occ_tensor = 0.5*np.ones_like(fd_tensor) - fd_tensor

	dw = ws[1]-ws[0]
	dkx = ( (kxs[1]-kxs[0])/float(len(kxs)) )
	dky = ( (kys[1]-kys[0])/float(len(kys)) )
	return np.mean(occ_tensor*A)*dw*len(ws) ### This should be the density 

######################
### Calculating Pi ###
######################
### This method returns a tensor of values of gamma_p[i,j]
def gen_A1g_tensor(kxs,kys,ws):
	kxv,kyv,wv = np.meshgrid(kxs,kys,ws)
	return A1g(kxv,kyv)

### This method returns a tensor of the FD function at the corresponding energies 
def gen_fd_tensor(kxs,kys,ws,mu,T):
	kxv,kyv,wv = np.meshgrid(kxs,kys,ws)

	return fd(wv,mu,T)

### Pi is computed as a convolution of the two spectral functions.
### We have imaginary part given by 
### Im Pi0[l,m,n] = 2piSt^2 int_{ijk} A[i,j,k] gamma[i,j]**2 A[i-l,j-m,k-n](f[k-n] - f[k])
### Im Pi1[l,m,n] = 2piSt^2 int_{ijk} A[i,j,k] gamma[i,j]*gamma[i-l,j-m,k-n]* A[i-l,j-m,k-n](f[k-n] - f[k])

### QUESTION: Is it ok to use mode='same' for the convolutions? 
### This has the effect of chopping the final convolution down to the size of the input arrays but it is unclear if this is the correct boundary condition for the momentum which should be periodic to capture Umklapp process
### It might be correct for frequency which should then be suffciently padded outside of the holes bandwidth
def calc_ImPi(kxs,kys,ws,A,mu,T):

	Nkx = len(kxs)
	Nky = len(kys)
	Nw = len(ws)
	dw = ws[1] - ws[0]
	#ImPi = np.zeros((2,Nkx,Nky,Nw))

	### First we construct the frist vector in the convolution
	### Form factor tensors  
	A1g_tensor = gen_A1g_tensor(kxs,kys,ws)
	fd_tensor = gen_fd_tensor(kxs,kys,ws,mu,T)

	### First we compute the term with f on the first spectral function 
	vec1 = A1g_tensor*A1g_tensor*A
	vec2 = fd_tensor*A
	vec3 = A1g_tensor*A1g_tensor*fd_tensor*A
	vec4 = A 

	conv_mode = 'same'
	
	### We want periodic boundary conditions for the convolutions but this will be challenging to implement in d = 3 
	### The native convolve2d method does not work but FFT convolve should work 

	ImPi0 = signal.fftconvolve(vec1,np.flip(vec2),mode=conv_mode) - signal.fftconvolve(vec3,np.flip(vec4),mode=conv_mode)
	ImPi0[:,:,:] *= 2.*np.pi*S*t**2*dw/float(Nkx*Nky) ### The momentum integrals are normalized by total number of points, energy by the differential

	### Now we repeat but for Pi1 we have slightly different form factor assignments in the convolution 

	### First we compute the term with f on the first spectral function 
	vec1 = A1g_tensor*A
	vec2 = fd_tensor*A1g_tensor*A

	ImPi1 = signal.fftconvolve(vec1,np.flip(vec2),mode=conv_mode) - signal.fftconvolve(vec2,np.flip(vec1),mode=conv_mode)
	ImPi1[:,:,:] *= 2.*np.pi*S*t**2*dw/float(Nkx*Nky) ### The momentum integrals are normalized by total number of points, energy by the differential

	return ImPi0,ImPi1

### This method will apply Kramers kronig relations to a function's imaginary part to obtain the retarded function 
### PiR[i,j,k] = 1./ (pi N) sum_l Im_part[i,j,l] 1./(ws[l] - ws[k] - i0^+) 
### This can make use of np.dot which sums the last axis of the first array with the (in this case first) axis of the second array
### Assumes frequency is the last argument of the tensor to be transformed
### We construct the PV of the real part using Hilbert transform and then simply use the old imaginary part such that we avoid numerical problems with using small eta
def Kramers_Kronig(ws,Im_part):
	### First we form the right Kramers Kronig tensor
	Nws = len(ws)
	dw = ws[1]-ws[0]
	kk_matrix_real = np.zeros((Nws,Nws),dtype=complex)
	for i in range(Nws):
		for j in range(Nws):
			#kk_matrix[i,j] = dw*((ws[j] - ws[i]) +1.j*zero)/(np.pi)*1./( (ws[j] - ws[i])**2 + zero**2) ### Sign is such that this will reconstruct the imaginary part to match the original imaginary part
			if i != j:
				kk_matrix_real[i,j] = dw/(np.pi)*1./(ws[j] - ws[i]) ### Sign is such that this will reconstruct the imaginary part to match the original imaginary part
	return Im_part@kk_matrix_real + 1.j*Im_part


########################################
### Computing Magnon Greens Funciton ###
########################################

### Accepts momentum and frequency and J and returns spin wave matrix kernel 
### returns (w tau_3 zeta_3 - K_q ) for w -> w+i0^+
### Shape is [4,4,Nkx,Nky,Nw] may be quite large 
def LSW_kernel(kxs,kys,ws,J):
	Nkx = len(kxs)
	Nky = len(kys)
	Nw = len(ws)

	kxvs,kyvs,wvs = np.meshgrid(kxs,kys,ws)
	
	wvs = wvs + 1.j*zero*np.ones_like(wvs)

	kernel = np.zeros((4,4,Nkx,Nky,Nw),dtype=complex)

	for i in range(4):
		if i < 2:
			kernel[i,i,...] = wvs*(-1)**i-4.*S*J*np.ones_like(wvs)
		if i >= 2:
			kernel[i,i] = -wvs*(-1)**i-4.*S*J*np.ones_like(wvs)

	a1g = gen_A1g_tensor(kxs,kys,ws)
	kernel[0,1] = -4.*S*J*a1g
	kernel[1,0] = -4.*S*J*a1g
	kernel[2,3] = -4.*S*J*a1g
	kernel[3,2] = -4.*S*J*a1g

	return kernel 

### This will include the Pi matrices in the computation of the magnon kernel 
### Returns same signature as bare kernel
### Pis are passed individually 
### We must be careful as they may not be symmetric in frequency or momentum 
def RPA_kernel(kxs,kys,ws,Pi0,Pi1,J):
	kernel = LSW_kernel(kxs,kys,ws,J)

	### Now we place the Pi components
	for i in range(2):
		kernel[i,i] += -Pi0 
		kernel[i,2] += -Pi1
		kernel[i,3] += -Pi1

	kernel[0,1] += -Pi0
	kernel[1,0] += -Pi0

	for i in range(2):
		kernel[2+i,2+i] += -np.flip(Pi0) ### We need -q components 
		kernel[2+i,0] += -Pi1
		kernel[2+i,1] += -Pi1

	kernel[2,3] += -np.flip(Pi0)
	kernel[3,2] += -np.flip(Pi0)

	return kernel

def main():

	T = 0.11*t
	U = 7.5*t
	J = 4.*t**2/U
	mu = -2*t ### Chemical potential 

	figDirectory = "../figures/"
	dataDirectory = "../data/"

	saveFigs = False

	holesDirectory = dataDirectory+"hole_spectra/03252025/"
	holesFile = "Hole_Spectral_functionJz0.0_alfa0.999_Nx20_Ny20"#"Hole_Spectral_functionJz0.05_alfa0.999_Nx20_Ny20"

	kxs,kys,ws,A = load_hole_spectrum(holesDirectory+holesFile)
	
	print("Hole doping: ",calc_density(kxs,kys,ws,A,mu,T))

	### Calculate RPA spectra from hole spectra 
	ImPi0,ImPi1 = calc_ImPi(kxs,kys,ws,A,mu,T)
	Pi0 = Kramers_Kronig(ws,ImPi0)
	Pi1 = Kramers_Kronig(ws,ImPi1)

	### Magnon propagators 
	magnon_kernel = RPA_kernel(kxs,kys,ws,Pi0,Pi1,J)
	magnon_propagator = np.zeros_like(magnon_kernel)

	for i in range(len(kxs)):
		for j in range(len(kys)):
			for k in range(len(ws)):
				magnon_propagator[:,:,i,j,k] = np.linalg.inv(magnon_kernel[:,:,i,j,k])


	### Plotting 


	### Frequency cuts of magnon propagator spectral function trace 
	indices = [ [0,0],[3,0],[6,0],[9,0],[12,0],[15,0]]
	clrs = cm.gist_heat(np.linspace(0.2,0.9,len(indices)))
	label_strings = [ r'$\mathbf{q}=($'+"{qx:0.2f}".format(qx=kxs[i[0]]/np.pi) + r'$\pi,$' + "{qy:0.2f}".format(qy=kys[i[1]]/np.pi) + r'$\pi)$'  for i in indices]
	
	if True:
		for j in range(len(indices)):
			i = indices[j]
			spec =np.abs( -1./np.pi* np.imag( np.trace( magnon_propagator[:,:,i[0],i[1],:] ) )/ws)
			plt.plot(ws/t,spec,label=label_strings[j],color=clrs[j])
			
		plt.xlabel(r'$\omega/t$')
		plt.xlim(-3.*J,3.*J)
		plt.yscale('log')
		plt.ylabel(r'$-\frac{1}{\pi}$Im tr$D(\omega,\mathbf{q})/t$')
		plt.legend()
		#if saveFigs: plt.savefig(figDirectory+"/Magnon_spectrum.pdf",bbox_inches='tight')
		plt.show()


	### Frequency momentum plot of magnon spectral function
	if True:
		spec = np.abs(-1./np.pi* np.imag( np.trace( magnon_propagator[:,:,:,0,:] ) )/ws)
		plt.imshow(np.transpose(spec),origin='lower',extent=[kxs[0],kxs[-1],ws[0],ws[-1]],aspect=0.8,cmap='coolwarm',norm=mclr.LogNorm())
		plt.ylabel(r'$\omega/t$')
		plt.ylim(-3.*J,3.*J)
		plt.xlabel(r'$q_x$')
		plt.xticks([0,np.pi/2.,np.pi,3.*np.pi/2.,2.*np.pi],[r'0',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
		plt.colorbar()
		#if saveFigs: plt.savefig(figDirectory+"/Magnon_spectrum.pdf",bbox_inches='tight')
		plt.show()

	quit()

	### Frequency dependence of Pi for kx cut 
	plt.imshow(np.transpose(ImPi0[:,0,:]),origin='lower',extent=[kxs[0],kxs[-1],ws[0],ws[-1]],aspect=0.4,cmap='coolwarm')
	plt.xlabel(r'$q_x$')
	plt.xticks([0,np.pi/2.,np.pi,3.*np.pi/2.,2.*np.pi],[r'0',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
	plt.ylabel(r'$\omega/t$')
	plt.colorbar()
	if saveFigs: plt.savefig(figDirectory+"/ImPi0_w_qx.pdf",bbox_inches='tight')
	plt.show()

	plt.imshow(np.transpose(ImPi1[:,0,:]),origin='lower',extent=[kxs[0],kxs[-1],ws[0],ws[-1]],aspect=0.4,cmap='coolwarm')
	plt.xlabel(r'$q_x$')
	plt.xticks([0,np.pi/2.,np.pi,3.*np.pi/2.,2.*np.pi],[r'0',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
	plt.ylabel(r'$\omega/t$')
	plt.colorbar()
	if saveFigs: plt.savefig(figDirectory+"/ImPi1_w_qx.pdf",bbox_inches='tight')
	plt.show()


	### Frequency cuts of magnon propagator spectral function trace 
	indices = [ [0,0],[10,0],[0,10],[10,10]]
	clrs = ['red','green','blue','purple']
	label_strings = [ r'$\mathbf{q}=($'+"{qx:0.0f}".format(qx=kxs[i[0]]/np.pi) + r'$\pi,$' + "{qy:0.0f}".format(qy=kys[i[1]]/np.pi) + r'$\pi)$'  for i in indices]
	

	### Frequency cuts of polarizations of magnons
	if True:
		for j in range(len(indices)):
			i = indices[j]
			plt.plot(ws/t,np.real(Pi0[i[0],i[1],:])/t,label=label_strings[j])
		
		plt.xlabel(r'$\omega/t$')
		plt.ylabel(r'Re$\Pi_0(\omega,\mathbf{q})/t$')
		plt.legend()
		if saveFigs: plt.savefig(figDirectory+"/RePi0.pdf",bbox_inches='tight')
		plt.show()

	if True:
		for j in range(len(indices)):
			i = indices[j]
			plt.plot(ws/t,np.imag(Pi0[i[0],i[1],:])/t,label=label_strings[j])
		
		plt.xlabel(r'$\omega/t$')
		plt.ylabel(r'Im$\Pi_0(\omega,\mathbf{q})/t$')
		plt.legend()
		if saveFigs: plt.savefig(figDirectory+"/ImPi0.pdf",bbox_inches='tight')
		plt.show()

	if True:
		for j in range(len(indices)):
			i = indices[j]
			plt.plot(ws/t,np.real(Pi1[i[0],i[1],:])/t,label=label_strings[j])
		
		plt.xlabel(r'$\omega/t$')
		plt.ylabel(r'Re$\Pi_1(\omega,\mathbf{q})/t$')
		plt.legend()
		if saveFigs: plt.savefig(figDirectory+"/RePi1.pdf",bbox_inches='tight')
		plt.show()

	if True:
		for j in range(len(indices)):
			i = indices[j]
			plt.plot(ws/t,np.imag(Pi1[i[0],i[1],:])/t,label=label_strings[j])
		
		plt.xlabel(r'$\omega/t$')
		plt.ylabel(r'Im$\Pi_1(\omega,\mathbf{q})/t$')
		plt.legend()
		if saveFigs: plt.savefig(figDirectory+"/ImPi1.pdf",bbox_inches='tight')
		plt.show()






if __name__ == "__main__":
	main()





