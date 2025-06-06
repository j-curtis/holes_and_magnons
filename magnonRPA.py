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


S = 0.5 ### Spin 1/2 
coord_z = 4. ### Coordination number for square lattice
t = 1. ### We will use units throughout with t = 1 for the time being 

rng = np.random.default_rng()

### some matrices useful for construction Nambu Bogoliubov matrices 
### we use matrices [0,1,2,3] to correspond to [identity, paul_x,pauli_y,pauli_z]
pauli = [ np.array([[1.,0.],[0.,1.]],dtype=complex), np.array([[0.,1.],[1.,0.]],dtype=complex),np.array([[0.,-1.j],[1.j,0.]],dtype=complex),np.array([[1.,0.],[0.,-1.]],dtype=complex) ]


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

### This method produces a fictitious hole spectrum which is a box in frequency and flat in momentum, for debugging purposes
def gen_box_A(kxs,kys,ws,W):
    Nkx = len(kxs)
    Nky = len(kys)
    Nw = len(ws)

    A = np.zeros((Nkx,Nky,Nw))
    for i in range(Nw):
        if ws[i] > -(W/2.) and ws[i]<(W/2.):
            A[:,:,i] = 1./W*np.ones((Nkx,Nky))
    return A

def gen_semicircle_A(kxs,kys,ws,W): 
    kxvs,kyvs,wvs = np.meshgrid(kxs,kys,ws,indexing='ij')
    out = np.zeros
    return 8./(np.pi*W**2)*np.real( np.sqrt((W/2.)**2 - (wvs)**2+0.j))

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

### This method generates the appropriate frequency grid for the magnon
### If the original frequency grid is odd and centered on zero this is the same
### If it is even it generates a grid which is centered and therefore one entry larger
### In this case the step size is preserved which means previously the frequency array was [-W...,-dw/2,dw/2,...,W] it is now [-W-dw/2...,-dw,0,dw,...]
def gen_magnon_freqs(ws):
	dw = ws[1] - ws[0]
	Nw = len(ws)

	if Nw %2 == 0:
		ws = np.concatenate((ws,[ws[-1]+dw]))-dw/2.

	return ws

### Home built autocorrelation method that is fast for 3D arrays and also respects periodic boundary conditions
### This is meant to numerically compute the integral h(q) = int_p f(p) g(p+q)  (does not normalize but computes the sum otherwise)
### FIRST TWO AXES: Performs standard circular autocorrelation 
### LAST AXIS: First pads with zeros then performs circular autocorrelation. Then shifts zero frequency to center. Then chops down back to size
### On the last axis it will by construction return an odd sized array which is either the same array as the frequencies passed (if already odd) or one size larger and is symmetric around zero frequency
def correlator_PBC(f,g):
    """Accepts arrays f,g w/shape (Nx,Nky,Nw) and returns the circular autocorrelation with automatically zero-padded last axis.
    Precisely this retuns z[i,j,k] = sum_{l,m,n} x[l,m,n] * y[i+l,j+m,k+n] padded appropriately on last axis.
    Finally, this shifts the frequencies to the center again at the end. 
    If number of passed frequencies is even this will generate an extra frequency point at zero to make the result Nw+1.
    """
    pad_size = f.shape[-1]
    
    f_padded = np.pad(f,( (0,0),(0,0),(pad_size,pad_size) ) ) 
    g_padded = np.pad(g,( (0,0),(0,0),(pad_size,pad_size) ) )  

    ### Now we take the FFT of both arrays 
    f_fft = np.fft.ifftn(f_padded,norm='forward')
    g_fft = np.fft.ifftn(g_padded,norm='forward')
    
    ### Next we take an element-wise product 
    ### conjugate to compute correlation instead of convolution
    h_fft = np.conjugate(f_fft)*g_fft

    ### Now we transform back
    h_padded = np.fft.fftn(h_fft,norm='forward')
    
    h = np.fft.fftshift(h_padded,axes=-1)

    if pad_size % 2 ==0 :
        h = h[...,pad_size:(h_padded.shape[-1]-pad_size+1)] ### We chop to get odd size with zero in middle

    else:
        h = h[...,pad_size:(h_padded.shape[-1]-pad_size)]

    return np.real(h)

### This method returns a tensor of values of gamma_p[i,j]
def gen_A1g_tensor(kxs,kys,ws):
	kxv,kyv,wv = np.meshgrid(kxs,kys,ws,indexing='ij')
	return A1g(kxv,kyv)

### This method returns a tensor of the FD function at the corresponding energies 
def gen_fd_tensor(kxs,kys,ws,mu,T):
	kxv,kyv,wv = np.meshgrid(kxs,kys,ws,indexing='ij')

	return fd(wv,mu,T)

### Pi is computed as a convolution of the two spectral functions.
### There are four functions which we compute directly as a matrix: Pi_{00} Pi_{01} Pi_{10} Pi_{11}
### ImPi[0,0,qx,qy,w] = -2 pi S z^2 t^2 int_{px,py,e} gamma[px,py]^2 A[px,py,e] A[px+qx,py+qy,e+w](f[e+w] - f[e])
### ImPi[0,1,qx,qy,w] = -2 pi S z^2 t^2 int_{px,py,e} gamma[px,py] gamma[px+qx,py+qy]  A[px,py,e] A[px+qx,py+qy,e+w](f[e+w] - f[e])
### ImPi[1,0,qx,qy,w] = -2 pi S z^2 t^2 int_{px,py,e} gamma[px,py] gamma[px+qx,py+qy]  A[px,py,e] A[px+qx,py+qy,e+w](f[e+w] - f[e])
### ImPi[1,1,qx,qy,w] = -2 pi S z^2 t^2 int_{px,py,e} gamma[px+qx,py+qy]^2  A[px,py,e] A[px+qx,py+qy,e+w](f[e+w] - f[e])
def calc_ImPi(kxs,kys,ws,A,mu,T):

	### In case it is not already properly formatted we will generate the correct magnon frequency grid 
	ws_out = gen_magnon_freqs(ws)

	Nkx = len(kxs)
	Nky = len(kys)
	Nw = len(ws)
	Nw_out = len(ws_out)

	dw = ws[1] - ws[0]

	ImPi = np.zeros((2,2,Nkx,Nky,Nw_out))

	### First we construct the frist vector in the convolution
	### Form factor tensors  
	A1g_tensor = gen_A1g_tensor(kxs,kys,ws)
	fd_tensor = gen_fd_tensor(kxs,kys,ws,mu,T)
	
	tensor_00 = A
	tensor_01 = fd_tensor*A 
	tensor_10 = A1g_tensor*A
	tensor_11 = A1g_tensor*fd_tensor*A 
	tensor_20 = A1g_tensor*A1g_tensor*A
	tensor_21 = A1g_tensor*A1g_tensor*fd_tensor*A 


	### Use convolution method of choice 
	### Here we will use the home built method
	convolver = correlator_PBC ### This is the convolution function we will call

	### This convolver does not normalize the sum (so we must multiple by appropriate normalizations) 
	### It should be the case that (up to step size) convolver(f,g) -> int_x f(x) g(x+y) 
 
	ImPi[0,0,...] = convolver(tensor_20, tensor_01) - convolver(tensor_21, tensor_00) 
	ImPi[0,1,...] = convolver(tensor_10, tensor_11) - convolver(tensor_11, tensor_10)
	ImPi[1,0,...] = convolver(tensor_10, tensor_11) - convolver(tensor_11, tensor_10)
	ImPi[1,1,...] = convolver(tensor_00, tensor_21) - convolver(tensor_01, tensor_20)

	ImPi *= -2.*np.pi*S*coord_z**2*t**2*dw/float(Nkx*Nky) ### The momentum integrals are normalized by total number of points, energy by the differential

	return ImPi

### This method is the analytically expected ImPi0 and ImPi1 for box DOS model
def box_ImPi(kxs,kys,ws,W,mu):

	### In case it is not already properly formatted we will generate the correct magnon frequency grid 
	ws = gen_magnon_freqs(ws)

	Nkx = len(kxs)
	Nky = len(kys)
	Nw = len(ws)

	dw = ws[1] - ws[0]
	ImPi = np.zeros((2,Nkx,Nky,Nw))
	a1g = gen_A1g_tensor(kxs,kys,ws)

	for i in range(Nw):
		w = ws[i]
		prefactor = -2.*np.pi*S*coord_z**2*t**2/(4.*W**2)
		Emax = min([W/2.-np.abs(w),mu])
		Emin = max([mu-np.abs(w),-W/2.])

		ImPi[0,...,i] = prefactor*(Emax-Emin)*float(Emax > Emin)*np.sign(w)

	ImPi[1,...] = ImPi[0,...]*a1g

	return ImPi

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
			if i != j:
				kk_matrix_real[i,j] = -dw/(np.pi)*1./(ws[j] - ws[i]) ### Sign is such that this will reconstruct the imaginary part to match the original imaginary part
	return Im_part@kk_matrix_real + 1.j*Im_part

### Same as above method but uses pre-computed Kramers-Kronig matrix to save on calculation time
def Kramers_Kronig_precomputed(kk_matrix,Im_part):
	return Im_part@kk_matrix + 1.j*Im_part

### This method computes the relevant KK matrix for the particular frequency bins 
def KK_matrix(ws):

	### Should be generated for the magnon frequency lattice 
	ws = gen_magnon_freqs(ws)

	### First we form the right Kramers Kronig tensor
	Nws = len(ws)
	dw = ws[1]-ws[0]

	kk_matrix_real = np.zeros((Nws,Nws),dtype=complex)
	for i in range(Nws):
		for j in range(Nws):
			if i != j:
				kk_matrix_real[i,j] = -dw/(np.pi)*1./(ws[j] - ws[i]) ### Sign is such that this will reconstruct the imaginary part to match the original imaginary part
	
	return kk_matrix_real

########################################
### Computing Magnon Greens Funciton ###
########################################

### Accepts momentum and frequency and J and returns spin wave matrix kernel 
### returns (w tau_3 - K_q ) for w -> w+i0^+
### Shape is [2,2,Nkx,Nky,Nw] 
def LSW_kernel(kxs,kys,ws,J):

	### To be safe we recast the magnon frequencies in to the right size 
	ws = gen_magnon_freqs(ws)


	Nkx = len(kxs)
	Nky = len(kys)
	Nw = len(ws)

	kxvs,kyvs,wvs = np.meshgrid(kxs,kys,ws,indexing='ij')
	
	dw = ws[1] - ws[0]
	### We adaptively choose broadening 

	zero = 2.*dw 
	wvs = wvs + 1.j*zero*np.ones_like(wvs) 

	a1g = gen_A1g_tensor(kxs,kys,ws)
	
	kernel = np.tensordot(pauli[3],wvs,axes=0) - coord_z*J*S*( np.tensordot(pauli[0],np.ones_like(wvs),axes=0)  + np.tensordot(pauli[1],a1g,axes=0) )

	return kernel 

### This will include the Pi matrices in the computation of the magnon kernel 
### Returns same signature as bare kernel
### Pis are passed individually 
### We must be careful as they may not be symmetric in frequency or momentum 
def RPA_kernel(kxs,kys,ws,Pi,J):
	kernel = LSW_kernel(kxs,kys,ws,J) 
	
	return kernel - Pi ### We can just add the kernel as it already has the correct matrix form 

### This returns the RPA propagator
def RPA_propagator(kxs,kys,ws,Pi,J):
	kernel = RPA_kernel(kxs,kys,ws,Pi,J)
	propagtor = np.zeros_like(kernel)

	for i in range(len(kxs)):
		for j in range(len(kys)):
			for k in range(len(ws)):
				propagtor[:,:,i,j,k] = np.linalg.inv(kernel[:,:,i,j,k])

	return propagtor

### This returns the RPA propagator
### The matrix is 2x2 so we can greatly accelerate by inverting by hand 
def RPA_propagator_analytic(kxs,kys,ws,Pi,J):
	kernel = RPA_kernel(kxs,kys,ws,Pi,J)
	propagtor = np.zeros_like(kernel)

	det = kernel[0,0,...]*kernel[1,1,...] - kernel[0,1,...]*kernel[1,0,...]
	propagtor[0,0,...] = kernel[1,1,...]/det 
	propagtor[0,1,...] = -kernel[0,1,...]/det 
	propagtor[1,0,...] = -kernel[1,0,...]/det 
	propagtor[1,1,...] = kernel[0,0,...]/det 

	return propagtor

### This takes the propagator and extracts the spectral function 
### We define this to be -1./(2pi) trIm  tau_3 DR
def RPA_spectrum(kxs,kys,ws,DRPA):
	return -1./(2.*np.pi)*np.imag( DRPA[0,0,...] - DRPA[1,1,...] ) ### should sum_{a,b} tau_3[a,b] D^R[b,a,i,j,k])

######################################
### For demler_tools compatibility ###
######################################

### This method is designed to be implemented using the demler_tools library from Radu
### It will take as input:
### 	location of the pickled files for the hole spectral functions
###		Temperature T
### 	Chemical potential mu
### 	Magnon superexchange J 
### It will then save as a pickled file the magnon RPA Greens function as well as the Pi bubbles in the designated location along with arrays of omega and k points used 
### The pickled output will be [kxs,kys,ws,propagator,Pi]
def compute_magnon_propagator(save_filename,hole_filename,T,mu,J):
	### First we load in the hole spectral functions 
	kxs,kys,ws,A = load_hole_spectrum(hole_filename)

	ws = gen_magnon_freqs(ws)

	### Next we compute the doping 
	delta = calc_density(kxs,kys,ws,A,mu,T)

	### Compute the imaginary parts of RPA Pi functions 
	ImPi = calc_ImPi(kxs,kys,ws,A,mu,T)

	### Kramers Kronig
	Pi = Kramers_Kronig(ws,ImPi)

	### Now construct the propagator 
	magnon_propagator = RPA_propagator(kxs,kys,ws,Pi,J)

	### Now we save the files 
	with open(save_filename,'wb') as savefile:
		pickle.dump((kxs,kys,ws,magnon_propagator,Pi),savefile)

	return None

############################################################
### Set of tools for handling fermion spectral functions ###
############################################################
class fermion_spectra:
	"""Set of methods for generating fermionic spectral functions"""

	def __init__(self):

		### These parameters set the spectral function array sizes 
		self.Nkx = None
		self.Nky = None 
		self.Nw = None 

		### Arrays of frequencies and momenta for spectral functions 
		self.kxs = None
		self.kys = None
		self.ws = None

		### Meshgrids of arguments 
		self.kx_grid = None
		self.ky_grid = None
		self.w_grid = None

		### The spectral function will be contained in here as a tensor 
		self.A = None

		### Metadata 
		self.type = None ### This is a flag which indicates what kind of spectral function we have, descriptively

	### This method generates a spectral function with flat semicircular in frequency shape 
	def generate_semicircle(self,Nkx,Nky,Nw,wmax,W):

		### These parameters set the spectral function array sizes 
		self.Nkx = Nkx
		self.Nky = Nky 
		self.Nw = Nw 

		### Now we generate arrays 
		self.kxs = np.linspace(0.,2.*np.pi,Nkx,endpoint=False)
		self.kys = np.linspace(0.,2.*np.pi,Nky,endpoint=False)
		self.ws = np.linspace(-wmax,wmax,Nw)
		self.dw = ws[1]-ws[0]

		### Now format as tensor meshgrids 
		self.kx_grid,self.ky_grid,self.w_grid = np.meshgrid(self.kxs,self.kys,self.ws,indexing='ij')

		### The spectral function will be contained in here as a tensor 
		self.A = 8./(np.pi*W**2)*np.real( np.sqrt((W/2.)**2 - (self.w_grid)**2+0.j))

		self.type = 'semicircle'
		self.W = W 

	### This method generates a spectral function in the shape of the YRZ ansatz 
	### Based on parameterization from James, Konik, Rice PRB 86 100508 (2012)
	### G^-1 = omega - xi - Delta^2/(omega + xi) where xi = -4t gamma_k  and Delta = delta_0 (coskx - cos ky )
	def generate_YRZ(self,Nkx,Nky,Nw,wmax,t,Delta_RVB,eta):
		### These parameters set the spectral function array sizes 
		self.Nkx = Nkx
		self.Nky = Nky 
		self.Nw = Nw 

		### Now we generate arrays 
		self.kxs = np.linspace(0.,2.*np.pi,Nkx,endpoint=False)
		self.kys = np.linspace(0.,2.*np.pi,Nky,endpoint=False)
		self.ws = np.linspace(-wmax,wmax,Nw)
		self.dw = ws[1]-ws[0]

		### Now format as tensor meshgrids 
		self.kx_grid,self.ky_grid,self.w_grid = np.meshgrid(self.kxs,self.kys,self.ws,indexing='ij')

		### The spectral function will be contained in here as a tensor 
		xi = -4.*self.t*A1g(self.kx_grid,self.ky_grid)
		delta = Delta_RVB*B2g(self.kx_grid,self.ky_grid)

		G = ( self.w_grid + 1.j*self.eta*np.ones_like(self.w_grid) - xi - (delta)**2/(self.w_grid + 1.j*eta*np.ones_like(self.w_grid) + xi ) )**(-1)

		self.A = G2A(G)

		self.type = 'yrz'
		self.t = t 
		self.Delta_RVB = Delta_RVB
		self.eta = eta 

	### This method will load a numerically computed spectral function from the specified file location 
	def load_hole_spectrum(self,hole_filename):

		with open(hole_filename,'rb') as f:
			kxs,kys,ws,G = pkl.load(f)

		self.type = 'numerical'
		self.hole_filename = hole_filename

		self.kxs = kxs
		self.kys = kys 
		self.ws = np.real(ws) ### sometimes the frequencies are passed as a complex array 
		self.dw = ws[1]-ws[0]

		self.Nkx = len(self.kxs)
		self.Nky = len(self.kys)
		self.Nw = len(self.ws)

		self.kx_grid,self.ky_grid,self.w_grid = np.meshgrid(self.kxs,self.kys,self.ws,indexing='ij')
		self.A = np.abs(G2A(G)) ### We need to take abs of spectral function which is directly computed form time-ordered G and needs to be converted to retarded

	### This method computes the doping density as a function of chemical potential and temperature 
	def calc_doping(self,mus = None,Ts=None):
		### We compute the doping vs chemical potential for each mu in the passed list and given temperature
		### If no mus are passed we do it for each frequency step 
		if (mus == None).any():
			mus = self.ws.copy()

		if (Ts == None).any():
			Ts = np.array([0.001*t]) ### default to just zero temperature 

		self.mus = mus
		self.Ts = Ts 

		### Now we encode in a meshgrid 
		self.mu_grid, self.T_grid = np.meshgrid(mus,Ts,indexing='ij')
		self.dopings = np.zeros_like(self.mu_grid)

		for i in range(len(mus)):
			for j in range(len(Ts)):
				fd_tensor = gen_fd_tensor(self.kxs,self.kys,self.ws,self.mu_grid[i,j],self.T_grid[i,j])

				occ_tensor = 0.5*np.ones_like(fd_tensor) - fd_tensor

				self.dopings[i,j] = np.mean(occ_tensor*self.A)*self.dw*float(self.Nw) ### This should be the density 

	### A simple plotting method 
	def plot_spectrum(self,bounds):
		### First we find a good frequency range 
		extents = [self.kxs[0],self.kxs[-1],self.ws[0],self.ws[-1]]

		plt.imshow(np.transpose(self.A[:,0,:]),extent=extents,origin='lower',cmap='magma',aspect=1.5)
		plt.colorbar()
		plt.ylim(bounds[0],bounds[1])
		plt.xlabel(r'$k_x$')
		plt.ylabel(r'$\omega/t$')
		plt.xticks([0,np.pi/2.,np.pi,3.*np.pi/2.,2.*np.pi],[r'0',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
		plt.show()

#####################################
### Set of tools for time-ordered ###
#####################################
class time_ordered:

	def __init__(self):
		self.kx = None ### momenta
		self.ky = None 
		self.w = None ### frequencies
		self.G = None ### Time-ordered GF of hole 
		self.A = None ### Spectral function of hole

		self.dw = None 

		self.mu = None 
		self.J = None ### LSWT superexchange J 
	
	### This method loads the time-orderd GF 
	@classmethod 
	def load_G(cls,hole_filename):
		with open(hole_filename,'rb') as f:
			kx,ky,w,G = pkl.load(f)


		return kx,ky,np.real(w),G

	### This method convers the retarded G to a TO G
	@classmethod 
	def G_R2TO(cls,kxs,kys,ws,G):
		G_r = np.real(G)
		G_i = np.imag(G)

		kxvs,kyvs,wvs = np.meshgrid(kxs,kys,ws)

		sign_w= np.sign(wvs)

		G_TO = G_r + 1.j*sign_w*G_i

		return G_TO

	### This method introduces a chemical potential to the Green's function which otherwise has no mu 
	@classmethod
	def add_mu(cls,ws,G,mu):
		### If the time-ordered Green's function is for mu = 0 we can get the finite mu case by flipping the sign of the imaginary part at E = mu  

		G_re = np.real(G)
		G_im = np.abs(np.imag(G))
		G_im[...,ws>mu] *= -1. ### Flip the sign of the imaginary part for frequencies less than chemical potential 


		return G_re + 1.j*G_im
	
	### This method computes the density from the spectral function 
	@classmethod
	def calc_density(cls,kxs,kys,ws,G,mu):
		A = 1./np.pi*np.abs(np.imag(G)) ### spectral function from TO GF 

		kxvs,kyvs,wvs = np.meshgrid(kxs,kys,ws)

		dw = ws[1]-ws[0]
		Nkx = len(kxs)
		Nky = len(kys)

		f = np.ones_like(wvs)
		f[wvs>mu]=0. ### Assign zeros to all entries above zero -- chemical potential is already included in the GF 

		delta = np.sum(A*f)*dw/float(Nkx*Nky)
		return delta 

	### This will compute the sum of the form int_x f(x)g(x+y) for complex functions f,g
	@classmethod
	def correlator_PBC(cls,f,g):
		"""Accepts arrays f,g w/shape (Nx,Nky,Nw) and returns the circular autocorrelation with automatically zero-padded last axis.
		First this computes h[i,j,k] = sum_{l,m,n} f[l,m,n] * g[i+l,j+m,k+n] padded appropriately on last axis.
		Finally, this shifts the frequencies to the center again at the end. 
		If number of passed frequencies is even this will generate an extra frequency point at zero to make the result Nw+1.
		"""
		pad_size = f.shape[-1]

		f_padded = np.pad(f,( (0,0),(0,0),(pad_size,pad_size) ) ) 
		g_padded = np.pad(g,( (0,0),(0,0),(pad_size,pad_size) ) )  

		### Now we take the FFT of both arrays 
		f_fft = np.fft.ifftn(f_padded,norm='forward')
		g_fft = np.fft.ifftn(g_padded,norm='forward')

		### Next we take an element-wise product 
		### conjugate to compute correlation instead of convolution
		h_fft = np.conjugate(f_fft)*g_fft

		### Now we transform back
		h_padded = np.fft.fftn(h_fft,norm='forward')

		h = np.fft.fftshift(h_padded,axes=-1)

		if pad_size % 2 ==0 :
		    h = h[...,pad_size:(h_padded.shape[-1]-pad_size+1)] ### We chop to get odd size with zero in middle

		else:
		    h = h[...,pad_size:(h_padded.shape[-1]-pad_size)]

		return h

	### This method computes the integral of the two Greens functions using the convolution method 
	@classmethod
	def calc_Pi(cls,kxs,kys,ws,G,mu):
		### First we introduce the chemical potential 
		G = cls.add_mu(ws,G,mu)
		### First we construct the proper frequency arrays 
		### In case it is not already properly formatted we will generate the correct magnon frequency grid 
		ws_out = gen_magnon_freqs(ws)

		Nkx = len(kxs)
		Nky = len(kys)
		Nw = len(ws)
		Nw_out = len(ws_out)

		dw = ws[1] - ws[0]

		Pi = np.zeros((2,2,Nkx,Nky,Nw_out),dtype=complex)

		### First we construct the frist vector in the convolution
		### Form factor tensors  
		a1g = gen_A1g_tensor(kxs,kys,ws)

		prefactor = -2.j*S*coord_z**2*t**2*dw/float(Nkx*Nky)

		### Now we compute the convolutions 
		### This will just be modified slightly to allow for complex functions 
		correlator = cls.correlator_PBC

		Pi[0,0,...] = prefactor*correlator(a1g**2*G,G)
		Pi[0,1,...] = prefactor*correlator(a1g*G,a1g*G)
		Pi[1,0,...] = prefactor*correlator(a1g*G,a1g*G)
		Pi[1,1,...] = prefactor*correlator(G,a1g**2*G)


		return Pi 






if __name__ == "__main__":
	main()





