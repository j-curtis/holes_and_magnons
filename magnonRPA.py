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
### There are two functions: Pi0 and Pi1. These have imaginary parts 
### CORRECTED -- I THINK ImPi[0,qx,qy,w] = -2 pi S z^2 t^2 int_{px,py,e} gamma[px+qx,py+qy]^2 A[px,py,e] A[px+qx,py+qy,e+w](f[e+w] - f[e])
### ImPi[0,qx,qy,w] = -2 pi S z^2 t^2 int_{px,py,e} gamma[px,py]^2 A[px,py,e] A[px+qx,py+qy,e+w](f[e+w] - f[e])
### ImPi[1,qx,qy,w] = -2 pi S z^2 t^2 int_{px,py,e} gamma[px,py] gamma[px+qx,py+qy]  A[px,py,e] A[px+qx,py+qy,e+w](f[e+w] - f[e])
def calc_ImPi(kxs,kys,ws,A,mu,T):

	### In case it is not already properly formatted we will generate the correct magnon frequency grid 
	ws_out = gen_magnon_freqs(ws)

	Nkx = len(kxs)
	Nky = len(kys)
	Nw = len(ws)
	Nw_out = len(ws_out)

	dw = ws[1] - ws[0]

	ImPi = np.zeros((2,Nkx,Nky,Nw_out))

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
 
	ImPi[0,...] = convolver(tensor_20, tensor_01) - convolver(tensor_21, tensor_00) 
	#ImPi[0,...] = convolver(tensor_00, tensor_21) - convolver(tensor_01, tensor_20) 
	ImPi[1,...] = convolver(tensor_10, tensor_11) - convolver(tensor_11, tensor_10)

	#ImPi[0,...] = np.flip( convolver(np.flip(tensor_00),tensor_21) - convolver(np.flip(tensor_01), tensor_20) )
	#ImPi[1,...] = np.flip( convolver(np.flip(tensor_10),tensor_11) - convolver(np.flip(tensor_11),tensor_10) )


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
				kk_matrix_real[i,j] = dw/(np.pi)*1./(ws[j] - ws[i]) ### Sign is such that this will reconstruct the imaginary part to match the original imaginary part
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
				kk_matrix_real[i,j] = dw/(np.pi)*1./(ws[j] - ws[i]) ### Sign is such that this will reconstruct the imaginary part to match the original imaginary part
	
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

	zero = 3.*dw 
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
	kernel[0,0,...] += - Pi[0,...] ### This should be Pi0^R(q)
	kernel[0,1,...] += - Pi[1,...] ### This should be Pi1^R(q)
	kernel[1,0,...] += - np.conjugate(np.flip(Pi[1,...])) ### This is Pi1^A(-q) and in principle this should be the same as Pi1^R(q) 
	kernel[1,1,...] += - np.conjugate(np.flip(Pi[0,...])) ### This should be Pi0^A(-q) 

	return kernel

### This returns the RPA propagator
def RPA_propagator(kxs,kys,ws,Pi,J):
	kernel = RPA_kernel(kxs,kys,ws,Pi,J)
	propagtor = np.zeros_like(kernel)

	for i in range(len(kxs)):
		for j in range(len(kys)):
			for k in range(len(ws)):
				propagtor[:,:,i,j,k] = np.linalg.inv(kernel[:,:,i,j,k])

	return propagtor

### This takes the propagator and extracts the spectral function 
### We define this to be -1./(2pi) trIm  tau_3 DR
def RPA_spectrum(kxs,kys,ws,DRPA):
	return -1./(2.*np.pi)*np.imag( np.trace( np.tensordot( pauli[3], DRPA, axes=1 ) ) ) ### should sum_{a,b} tau_3[a,b] D^R[b,a,i,j,k])

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

def main():
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

	T = 0.11*t
	U = 7.5*t
	J = 4.*t**2/U
	mu = -1*t ### Chemical potential 

	figDirectory = "../figures/"
	dataDirectory = "../data/"

	saveFigs = False

	holesDirectory = dataDirectory+"hole_spectra/03252025/"
	holesFile = "Hole_Spectral_functionJz0.05_alfa0.999_Nx20_Ny20"#"Hole_Spectral_functionJz0.05_alfa0.999_Nx20_Ny20"

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
			spec =np.abs( -1./np.pi* np.imag( np.linalg.det( magnon_propagator[:,:,i[0],i[1],:],axes=[0,1] ) )/ws)
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





