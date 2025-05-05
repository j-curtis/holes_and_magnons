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
zero = 1.e-5*t ### Infinitesimal broadening factor used to regularize singular magnon kernel. Hopefully should be smaller tha contribution due to holes 

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


### Home built convolution that is fast for 3D arrays and also respects periodic boundary conditions
### Also it zero pads on last (energy) axis so that artifacts are removed 
### Arrays may be complex
def convolve_PBC(x,y,pad=0):
    if pad >0 :
        x = np.pad(x,((0,0),(0,0),(pad,pad)))
        y = np.pad(y,((0,0),(0,0),(pad,pad)))
    
    ### First we take an FFT of both arrays 
    x_fft = np.fft.fftn(x)
    y_fft = np.fft.fftn(y)

    ### Next we take an element-wise product 

    z_fft = x_fft*y_fft 

    ### Now we transform back
    z = np.fft.ifftn(z_fft)
    z = np.fft.ifftshift(z,axes=-1)
    ### We also have to shift since numpy shifts the fft to nonsymmetric interval
    last_axis_size=z.shape[-1]
    z = z[...,pad:(last_axis_size-pad)] ### Chop back down on the last axis 
    return np.real(z) 

### This method returns a tensor of values of gamma_p[i,j]
def gen_A1g_tensor(kxs,kys,ws):
	kxv,kyv,wv = np.meshgrid(kxs,kys,ws,indexing='ij')
	return A1g(kxv,kyv)

### This method returns a tensor of the FD function at the corresponding energies 
def gen_fd_tensor(kxs,kys,ws,mu,T):
	kxv,kyv,wv = np.meshgrid(kxs,kys,ws,indexing='ij')

	return fd(wv,mu,T)

### Pi is computed as a convolution of the two spectral functions.
### We have imaginary part given by 
### Im Pi[l,m,n] = -1/2 pi S z^2 t^2 int_{ijk} ( gamma[i,j] + gamma[i+l,j+m] )^2 A[i,j,k] A[i+l,j+m,k+n](f[k+n] - f[k])

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
	### This will be a sum of three convolutions which come from expanding out the square of the form factor 
	
	### (A1g * A1g *A ) . (A * FD) + 2. * ( A1g *A ) . (A1g *A * FD) + A. (A1g*A1g*FD*A)
	### Then subtracted from this is the same with FD on the other side 
	### The relevant tensors are A, FD *A, A1g*A, A1g*A*FD, A1g*A1g*A,A1g*A1g*A*FD

	tensor_00 = A
	tensor_01 = fd_tensor*A 
	tensor_10 = A1g_tensor*A
	tensor_11 = A1g_tensor*fd_tensor*A 
	tensor_20 = A1g_tensor*A1g_tensor*A
	tensor_21 = A1g_tensor*A1g_tensor*fd_tensor*A 


	### Use convolution method of choice 
	### Here we will use the home built method
	convolver = convolve_PBC ### This is the convolution function we will call 
	pad = 400 ### We pad last axis just to be careful

	ImPi = convolver(tensor_20,np.flip(tensor_01),pad) + 2.*convolver(tensor_10,np.flip(tensor_11),pad) + convolver(tensor_00,np.flip(tensor_21),pad) 
	ImPi -= convolver(tensor_21,np.flip(tensor_00),pad) + 2.*convolver(tensor_11,np.flip(tensor_10),pad) + convolver(tensor_01,np.flip(tensor_20),pad)

	#ImPi = convolver(vec1,np.flip(vec2),pad) - convolver(vec3,np.flip(vec4),pad)
	ImPi[:,:,:] *= 0.5*np.pi*S*coord_z**2*t**2*dw/float(Nkx*Nky) ### The momentum integrals are normalized by total number of points, energy by the differential

	return ImPi

### This method is the analytically expected Pi
def box_Pi(kx,ky,w,W,mu):
    prefactor = -2.*np.pi*S*t**2/(4.*W**2)
    Emax = min([W/2.-np.abs(w),mu])
    Emin = max([mu-np.abs(w),-W/2.])
    return prefactor*(Emax-Emin)*float(Emax > Emin)*np.sign(w)

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

	kxvs,kyvs,wvs = np.meshgrid(kxs,kys,ws,indexing='ij')
	
	wvs = wvs + 1.j*zero*np.ones_like(wvs)

	kernel = np.zeros((2,2,Nkx,Nky,Nw),dtype=complex)

	a1g = gen_A1g_tensor(kxs,kys,ws)
	
	for i in range(2):
		for j in range(2):
			kernel[i,j,...] = pauli[3][i,j]*wvs - coord_z*J*S*pauli[0][i,j]*np.ones_like(wvs) - coord_z*J*S*pauli[1][i,j]*a1g

	return kernel 

### This will include the Pi matrices in the computation of the magnon kernel 
### Returns same signature as bare kernel
### Pis are passed individually 
### We must be careful as they may not be symmetric in frequency or momentum 
def RPA_kernel(kxs,kys,ws,Pi,J):
	kernel = LSW_kernel(kxs,kys,ws,J)

	### Now we place the Pi components
	#kernel[0,0,...] += - Pi
	#kernel[1,1,...] += - np.flip(Pi) ### q -> -q for this component
	#kernel[0,1,...] += -Pi 
	#kernel[1,0,...] += -np.conj(Pi)

	kernel += -np.tensordot( pauli[0] + pauli[1], Pi,axes=0)

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
def RPA_spectrum(kxs,kys,ws,DRPA):
	kxvs,kyvs,wvs = np.meshgrid(kxs,kys,ws,indexing='ij')
	### determinant wants last two axes the matrix axes
	mat = np.moveaxis(DRPA,[0,1],[-2,-1])
	return 1./np.pi*np.imag( np.linalg.det(mat) )*wvs

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
### The pickled output will be [kxs,kys,ws,propagator,Pi0,Pi1]
def compute_magnon_propagator(save_filename,hole_filename,T,mu,J):
	### First we load in the hole spectral functions 
	kxs,kys,ws,A = load_hole_spectrum(hole_filename)

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





