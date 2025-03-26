### Evaluates RPA propagators for magnons due to interaction with holes
### Jonathan Curtis 
### 03/22/25

import numpy as np
import pickle as pkl
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
zero = 1.e-5*t

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
def load_hole_spectrum(fpath):
	with open(fpath,'rb') as f:
		kxs,kys,ws,G = pkl.load(f)

	return kxs,kys,np.real(ws),G2A(G)


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
	return np.mean(occ_tensor*A)### This should be the density 


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

	mode_conv = 'same'

	ImPi0 = signal.convolve(vec1,np.flip(vec2),mode=mode_conv) - signal.convolve(vec3,np.flip(vec4),mode=mode_conv)
	ImPi0[:,:,:] *= 2.*np.pi*S*t**2/float(Nkx*Nky*Nw) ### The integrals are normalized by total number of points 

	### Now we repeat but for Pi1 we have slightly different form factor assignments in the convolution 

	### First we compute the term with f on the first spectral function 
	vec1 = A1g_tensor*A
	vec2 = fd_tensor*A1g_tensor*A

	ImPi1 = signal.convolve(vec1,np.flip(vec2),mode=mode_conv) - signal.convolve(vec2,np.flip(vec1),mode=mode_conv)
	ImPi1[:,:,:] *= 2.*np.pi*S*t**2/float(Nkx*Nky*Nw) ### The integrals are normalized by total number of points 

	return ImPi0,ImPi1

### This method will apply Kramers kronig relations to a function's imaginary part to obtain the retarded function 
### PiR[i,j,k] = 1./ (pi N) sum_l Im_part[i,j,l] 1./(ws[l] - ws[k] - i0^+) 
### This can make use of np.dot which sums the last axis of the first array with the (in this case first) axis of the second array
### Assumes frequency is the last argument of the tensor to be transformed
def Kramers_Kronig(ws,Im_part):
	### First we form the right Kramers Kronig tensor
	Nws = len(ws)
	kk_matrix = 1./(np.pi*float(Nws))*1./(np.tensordot(ws,np.ones_like(ws),axes=-1) - np.tensordot(np.ones_like(ws),ws,axes=-1) - 1.j*zero )
	return Im_part@kk_matrix

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

	kernel = np.zeros((4,4,Nkx,Nky,Nw),dtype=complex)

	for i in range(4):
		if i < 2:
			kernel[i,i,...] = (wvs+1.j*zero*np.ones_like(wvs))*(-1)**i-4.*S*J*np.ones_like(wvs)
		if i >= 2:
			kernel[i,i] = -(wvs+1.j*zero*np.ones_like(wvs))*(-1)**i-4.*S*J*np.ones_like(wvs)

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




### This gives the Bogoliubov magnon propagator as a matrix 
### The matrix elements are for the Bogoliubov spinor (a_q,b_-q^+, a_-q^+, b_q) components 



##########################################
### DEPRECATED METHODS FOR MONTE CARLO ###
##########################################
### This method uniformly samples the momentum points 
"""
def gen_kpts_uniform(ps):
	N = int(ps.NMC)
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

### Hole spectral function for particular energy and momentum 
### This will be obtained 
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

### This function evalutes the imaginary part of the RPA response function for a given magnon frequency and momentum  
### These are passed as q = [w,qx,qy]
def ImPi_MC(q,ps):

	### We need to evaluate the integrals 
	### integral_{E,p} iA(E,p) A(E+w,p+q) gamma_{p} [ gamma_{p}, gamma_{p+q} ] ( f(E)-f(E') )/(E - E'+w)

	### We start by Monte Carlo sampling the hole momentum and energy k,E according to the spectral function 

	N = int(ps.NMC)
	k1s = hole_sample(ps) ### This generates a sample of energies and momenta for holes 

	k2s = k1s + np.tensordot(q,np.ones(N),axes=-1) ### This is the sum of the energy and momenta 

	### The only part that differs is the form factor 

	integrands= np.zeros(N) 
	for i in range(N):
		integrands[i] = 2.*np.pi*S*ps.t**2*hole_spectrum(k2s[:,i],ps)*( 0.5*np.tanh(0.5*k1s[0,i]/ps.T) - 0.5*np.tanh(0.5*k2s[0,i]/ps.T) )

	means = np.array([ np.mean(integrands*(Ag(k1s[1:,:]))**2 ), np.mean(integrands*Ag(k1s[1:,:])*Ag(k2s[1:,:]) )  ])
	errs =  np.array([ np.std(integrands*(Ag(k1s[1:,:]))**2 ), np.std(integrands*Ag(k1s[1:,:])*Ag(k2s[1:,:]) )  ])

	return means, errs

### This function evalutes the imaginary part of the RPA response function for a given magnon frequency and momentum  
### These are passed as q = [w,qx,qy]
def ImPi_quad(q,ps):

	### We need to evaluate the integrals 
	### integral_{E,p} iA(E,p) A(E+w,p+q) gamma_{p} [ gamma_{p}, gamma_{p+q} ] ( f(E)-f(E') )/(E - E'+w)

	integrand_1 = lambda w,kx,ky: 2.*np.pi*S*ps.t**2*Ag(np.array([kx,ky]))**2*hole_spectrum(np.array([w,kx,ky]),ps)*hole_spectrum(np.array([w,kx,ky])+q,ps)*( 0.5*np.tanh(0.5*w/ps.T) - 0.5*np.tanh(0.5*(w+q[0])/ps.T) )
	integrand_2 = lambda w,kx,ky: 2.*np.pi*S*ps.t**2*Ag(np.array([kx,ky]))*Ag(np.array([kx,ky]) + q[1:])*hole_spectrum(np.array([w,kx,ky]),ps)*hole_spectrum(np.array([w,kx,ky])+q,ps)*( 0.5*np.tanh(0.5*w/ps.T) - 0.5*np.tanh(0.5*(w+q[0])/ps.T) )

	return np.array([ intg.tplquad(integrand_1,-np.pi,np.pi,-np.pi,np.pi,-ps.Emax,ps.Emax)[0]/(4.*np.pi**2) , intg.tplquad(integrand_2,-np.pi,np.pi,-np.pi,np.pi,-ps.Emax,ps.Emax)[0]/(4.*np.pi**2) ])
"""

def main():

	T = 0.11*t
	U = 7.5*t
	J = 4.*t**2/U

	figDirectory = "../figures/"
	dataDirectory = "../data/"

	holesDirectory = dataDirectory+"hole_spectra/03252025/"
	holesFile = "Hole_Spectral_functionJz0.0_alfa0.999_Nx20_Ny20"

	kxs,kys,ws,A = load_hole_spectrum(holesDirectory+holesFile)

	A = np.abs(A) ### Sign errors in spectral function 
	print(np.mean(A)) ### Should equal 1 if normalized 

	#plt.imshow(np.transpose(A[:,0,:]),origin='lower',extent=[kxs[0],kxs[-1],ws[0],ws[-1]],aspect=0.3)
	#plt.colorbar()
	#plt.show()

	nmus = 100
	mus = np.linspace(-20.*t,20.*t,nmus)
	deltas = np.zeros_like(mus)
	for i in range(nmus):
		deltas[i] = calc_density(kxs,kys,ws,A,mus[i],T)

	#plt.plot(mus,deltas)
	#plt.xlabel(r'$\mu/t$')
	#plt.ylabel(r'$\delta$')
	#plt.show()

	mu = -3.8*t
	ImPi0,ImPi1 = calc_ImPi(kxs,kys,ws,A,mu,T)
	Pi0 = Kramers_Kronig(ws,ImPi0)
	Pi1 = Kramers_Kronig(ws,ImPi1)

	magnon_kernel = LSW_kernel(kxs,kys,ws,J)
	print(magnon_kernel.shape)
	magnon_prop = np.zeros_like(magnon_kernel)

	for i in range(len(kxs)):
		for j in range(len(kys)):
			for k in range(len(ws)):
				magnon_prop[:,:,i,j,k] = np.linalg.inv(magnon_kernel[:,:,i,j,k])

	plt.imshow(np.transpose(-1./np.pi*np.imag(np.trace(magnon_prop[:,:,:,0,:]) )),origin='lower', extent=[kxs[0],kxs[-1],ws[0],ws[-1]],aspect=1.,cmap='coolwarm')
	plt.colorbar()
	plt.xlabel(r'$q_x$')
	plt.ylim(-2.*J,2.*J)
	plt.ylabel(r'$\omega/t$')
	plt.show()
	quit()


	indices = [ [0,0],[10,0],[10,10],[-1,10],[-1,-1] ]

	for i in indices:
		label_string = r'$\mathbf{q}=($'+"{qx:0.0f}".format(qx=kxs[i[0]]/np.pi) + r'$\pi,$' + "{qy:0.0f}".format(qy=kys[i[1]]/np.pi) + r'$\pi)$' 
		plt.plot(ws/t,np.real(Pi0[i[0],i[1],:])/t,label=label_string)
	
	plt.xlabel(r'$\omega/t$')
	plt.ylabel(r'Re$\Pi_0(\omega,\mathbf{q})/t$')
	plt.legend()
	plt.savefig(figDirectory+"/RePi0.pdf",bbox_inches='tight')
	plt.show()

	for i in indices:
		label_string = r'$\mathbf{q}=($'+"{qx:0.0f}".format(qx=kxs[i[0]]/np.pi) + r'$\pi,$' + "{qy:0.0f}".format(qy=kys[i[1]]/np.pi) + r'$\pi)$' 
		plt.plot(ws/t,np.imag(Pi0[i[0],i[1],:])/t,label=label_string)
	
	plt.xlabel(r'$\omega/t$')
	plt.ylabel(r'Im$\Pi_0(\omega,\mathbf{q})/t$')
	plt.legend()
	plt.savefig(figDirectory+"/ImPi0.pdf",bbox_inches='tight')
	plt.show()

	for i in indices:
		label_string = r'$\mathbf{q}=($'+"{qx:0.0f}".format(qx=kxs[i[0]]/np.pi) + r'$\pi,$' + "{qy:0.0f}".format(qy=kys[i[1]]/np.pi) + r'$\pi)$' 
		plt.plot(ws/t,np.real(Pi1[i[0],i[1],:])/t,label=label_string)
	
	plt.xlabel(r'$\omega/t$')
	plt.ylabel(r'Re$\Pi_1(\omega,\mathbf{q})/t$')
	plt.legend()
	plt.savefig(figDirectory+"/RePi1.pdf",bbox_inches='tight')
	plt.show()

	for i in indices:
		label_string = r'$\mathbf{q}=($'+"{qx:0.0f}".format(qx=kxs[i[0]]/np.pi) + r'$\pi,$' + "{qy:0.0f}".format(qy=kys[i[1]]/np.pi) + r'$\pi)$' 
		plt.plot(ws/t,np.imag(Pi1[i[0],i[1],:])/t,label=label_string)
	
	plt.xlabel(r'$\omega/t$')
	plt.ylabel(r'Im$\Pi_1(\omega,\mathbf{q})/t$')
	plt.legend()
	plt.savefig(figDirectory+"/ImPi1.pdf",bbox_inches='tight')
	plt.show()






if __name__ == "__main__":
	main()





