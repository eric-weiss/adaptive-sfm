import numpy as np
import theano
import theano.tensor as T
import cPickle as cp

from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict


class BilinearModel():
	
	def __init__(self, data_dims, c_dims, v_dims, image_var=1.0,
					c_var=1.0, v_var=1.0/25.0):
	
		self.data_dims=data_dims #dimensionality of the data
		self.c_dims=c_dims # dims. of the image representation
		self.v_dims=v_dims # dims. of the dynamical variables
		self.image_var=image_var # assumed variance of each pixel
		
		# The matrix W determines the mapping between the latent
		# varible c and the video frames, and M determines the dynamics
		# through time on c, as well as their interaction with v.
		# I initialize them randomly and make sure they're float32 
		# since theano only does fast GPU ops for float32
		init_W=np.random.randn(c_dims, data_dims).astype(np.float32)/10.0
		
		#For every element v_i of v there is a corresponding dynamical 
		#matrix M_i. Each of these matrices have shape
		#(c_dims, c_dims), and there are v_dims of them. I store them 
		#as a stack in a 3D tensor that has shape (v_dims, c_vims, c_dims),
		#so that M[i] will give you a matrix.
		init_M=np.random.randn(v_dims, c_dims, c_dims).astype(np.float32)/100.0
		
		#As for any variable that has a persistent state, i.e. model
		#parameters, you need to create theano shared variables. This
		#also tells theano that you might want to operate on these
		#variables using the GPU so it knows to copy them over
		self.W=theano.shared(init_W, 'W')
		self.M=theano.shared(init_M, 'M')
		
		#c_var and v_var represent the variances of the v and c 
		#through time (i.e. the variance of P(v_t+1 | v_t ) etc.)
		#We might want to learn these, hence the declaration as shared
		#variables. Since variance must be 
		#greater than zero, I parametrize the variance as the 
		#exponential of some other number, since exp(x)>0 for any x. 
		#(You never know what the learning algorithm might do to the 
		#parameters, so I'm playing it safe.)
		init_ln_v_var=(np.ones(v_dims)*np.log(v_var)).astype(np.float32)
		self.ln_v_var=theano.shared(init_ln_v_var,'ln_v_var')
		
		init_ln_c_var=(np.ones(c_dims)*np.log(c_var)).astype(np.float32)
		self.ln_c_var=theano.shared(init_ln_c_var,'ln_c_var')
		
		#I do this simply so we don't have to remember to take the 
		#exponential of ln_v_var or ln_c_var; instead we just use 
		#the variables below and theano will take care of the math
		self.v_var=T.exp(self.ln_v_var)
		self.c_var=T.exp(self.ln_c_var)
		
		#I'm using gradient descent optimization with momentum, so I
		#need to keep track of the momentum for each parameter
		self.dW=theano.shared(init_W*0.0)
		self.dM=theano.shared(init_M*0.0)
		self.dln_v_var=theano.shared(init_ln_v_var*0.0)
		self.dln_c_var=theano.shared(init_ln_c_var*0.0)
		
		#All the parameters that we intend to learn (and their momentums)
		#should be put inside these lists
		self.params=[self.W, self.M]#, self.ln_v_var, self.ln_c_var]
		self.momentums=[self.dW, self.dM]#, self.dln_v_var, self.dln_c_var]
	
	
	def compute_predicted_c(self, c, v):
		''' This function takes an nT-by-c_dims matrix c and
		an nT-by-v_dims matrix v and computes the predicted c_t+1 for 
		every c_t and v_t (i.e., the mean of P(c_t+1 | c_t, v_t) )
		'''		
		def step(M_i, v_i, totalsum, c):
			cMv=T.dot(c, M_i)*v_i.dimshuffle(0,'x')
			return cMv+totalsum
		
		#This loops over the elements of v and the corresponding Ms,
		#summing each of their contributions to the predicted c
		init_sum=c*0.0
		c_sums, updates = theano.scan(fn=step,
									sequences=[self.M, v.T],
									outputs_info=[init_sum],
									non_sequences=[c])
		
		c_pred=c_sums[-1]+c
		
		return c_pred
	
	
	def log_posterior(self, c, v, x):
		''' This function takes an nT-by-c_dims matrix c, 
		an nT-by-v_dims matrix v, and an nT-by-data_dims matrix x and
		returns the log posterior for each pair of adjacent points
		as an nT-1 dimensional vector.
		'''	
		
		#Each of these distributions is gaussian, hence the 
		#sums-of-squared-differences
		c_pred=self.compute_predicted_c(c, v)
		c_terms=-T.sum(((c[1:]-c_pred[:-1])**2)/(2.0*self.c_var.dimshuffle('x',0)),axis=1)
		
		recons=T.dot(c, self.W)
		image_terms=-T.sum((x-recons)**2,axis=1)/(2.0*self.image_var)
		
		v_terms=-T.sum(((v[1:]-v[:-1])**2)/(2.0*self.v_var.dimshuffle('x',0)),axis=1)
		
		log_p = c_terms + image_terms[1:] + v_terms
		
		return log_p
	
	
	def MAP_step(self, c, dc, v, dv, mu, x, stepsize):
		'''performs a step of gradient descent with Nesterov
		momentum on the negative log-posterior. It is used inside 
		a scan loop in the function get_MAP_estimate.
		c, v are matrices as usual
		dc, dv are their momentums (same shapes)
		mu is the momentum coefficient (between 0 and 1)
		x is the data, also a matrix
		stepsize is the gradient descent stepsize
		'''
		
		#An experiment I was trying - will cause the momentum coefficient
		#to grow during optimization but never reach 1.0.
		mu_rate=0.001
		new_mu=T.cast(mu_rate*(1.0-mu)+mu,'float32')
		
		c_test=c+mu*dc
		v_test=v+mu*dv
		
		#This term encourages v to be sparse without penalizing
		#its L2 norm. It's not really necessary for the model to work,
		#but it makes the inferred values of v easier to interpret.
		#(It should really be put inside of the function log_posterior)
		v_sparsity_prior=T.mean(T.abs_(v_test/T.sqrt(T.sum(v_test**2,axis=1)+1e-6).dimshuffle(0,'x')))*0.01
		
		#This is the total objective to be minimized
		objective=T.sum(-self.log_posterior(c_test,v_test,x) + v_sparsity_prior)
		
		#Taking gradients - I multiply them by their (fourth root)
		#variances to make it easier to set appropriate stepsizes
		cgrad=T.grad(T.sum(objective), c_test)*T.sqrt(T.sqrt(self.c_var)).dimshuffle('x',0)
		vgrad=T.grad(T.sum(objective), v_test)*T.sqrt(T.sqrt(self.v_var)).dimshuffle('x',0)
		
		#optimization math
		dv_new=mu*dv-stepsize*vgrad
		dc_new=mu*dc-stepsize*cgrad
		v_new=v+dv_new
		c_new=c+dc_new
		
		return c_new, dc_new, v_new, dv_new, new_mu, objective/T.cast(c_test.shape[0],'float32')
	
	
	def get_MAP_estimate(self, x, c0, v0, nsteps, stepsize, init_mu=0.5):
		
		#this function takes an nT-by-data_dims block of training data
		#and returns the MAP estimate for c and v as matrices
		
		dc0=c0*0.0
		dv0=v0*0.0
		
		[cs, dcs, vs, dvs, mus, objs], updates = theano.scan(fn=self.MAP_step,
								outputs_info=[c0, dc0, v0, dv0, T.cast(init_mu,'float32'),None],
								non_sequences=[x,T.cast(stepsize,'float32')],
								n_steps=nsteps)
		
		#Everything returned by the theano.scan is a list. The last 
		#values correspond to the MAP estimates for c and v so that's
		#all I return. I also return the entire history of the objective
		#(objs) to monitor the stability of the MAP optimization.
		return cs[-1], vs[-1], objs
	
	
	def update_params(self, x, lrate, stepsize, nsteps, c0, v0):
		
		#this function takes an nb-by-nx block of training samples, 
		#finds the MAP estimate for each of them, and does a step of 
		#gradient descent (with momentum) on the log-likelihood
		
		cs, vs, objs = self.get_MAP_estimate(x, c0, v0, nsteps, stepsize, 0.99)
		c_MAP=cs
		v_MAP=vs
		
		log_L=self.log_posterior(c_MAP, v_MAP, x)
		
		#This is a prior on M that encourages each M_i to be orthogonal
		#to each other, the idea being to prevent redundancy early
		#in training.
		M_flat=T.reshape(self.M,(self.v_dims,self.c_dims**2))
		M_cov=T.dot(M_flat,M_flat.T)*(T.eye(self.v_dims)-1.0)
		
		objective=-T.mean(log_L) + T.sum(T.abs_(M_cov))*0.0
		
		updates=OrderedDict()
		
		mu=0.99 #The momentum coefficient for the model parameters
		
		#To compute the learning update for each of the parameters I 
		#loop over self.params and self.momentums, calculating the 
		#gradients etc.
		for i in range(len(self.params)):
			param=self.params[i]
			mom=self.momentums[i]
			
			gparam=T.grad(objective, param, consider_constant=[c_MAP, v_MAP])
			step=mu*mom-lrate*gparam
			new_param=param+step
			
			if param==self.W:
				#Each column of W is constrained to be unit-length
				new_param=new_param/T.sqrt(T.sum(new_param**2,axis=1)).dimshuffle(0,'x')
			if param==self.M:
				#Each M_i is constrained to have a constant Frobenius 
				#norm - this might be overkill
				new_param=0.2*new_param/T.sqrt(T.sum(T.sum(new_param**2,axis=2),axis=1)).dimshuffle(0,'x','x')
			
			updates[param]=T.cast(new_param,'float32')
			updates[mom]=T.cast(step,'float32')
		
		#resid_var just measures how well the model is representing
		#the pixel data
		recons=T.dot(c_MAP, self.W) #"reconstructed data"
		resid_var=T.mean(T.sum((recons-x)**2,axis=1)/T.sum(x**2,axis=1))
		
		return cs, vs, objs, resid_var, updates

















