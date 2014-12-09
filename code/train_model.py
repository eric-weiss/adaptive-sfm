from matplotlib import pyplot as pp
import numpy as np
import theano
import theano.tensor as T
import cPickle as cp
from utils import tile_raster_images
import matplotlib.cm as cm
import scipy.optimize as sio

def lossfunc(x,r):
	return (1.0-x)**np.exp(r)


from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict

from models import BilinearModel

data_fn='../data/translations_2w16.cpl'
f=open(data_fn,'rb')
data_raw=cp.load(f)
f.close()
nt, nx=data_raw.shape

nc=10**2
nv=3

data=theano.shared(data_raw)

#pp.plot(data_raw[:,:])
#pp.show()

model=BilinearModel(nx, nc, nv, v_var=4e-3, image_var=1e-2, c_var=2e-3)

lr=T.fscalar() #learning rate
inimu=T.fscalar() #initial MAP momentum coefficient
c0T=T.fmatrix() #initial MAP c estimate
v0T=T.fmatrix() #initial MAP v estimate
nstepsT=T.iscalar() #number of steps to perform in MAP optimization

cs, vs, objs, resvar, learning_updates=model.update_params(data, lr, inimu, nstepsT, c0T, v0T)

#This function updates the model parameters and returns the MAP estimates
#for c and v, as well as the residual variance and the MAP objective
#history
learn=theano.function([lr, inimu, nstepsT, c0T, v0T],[cs, vs, resvar,objs],updates=learning_updates,allow_input_downcast=True)

#Plotting stuff
pp.ion()
fig=pp.figure(figsize=(12,6))
W=model.W.get_value()
wpic=tile_raster_images(W,(int(np.sqrt(nx)),int(np.sqrt(nx))),(int(np.sqrt(nc)),int(np.sqrt(nc))),tile_spacing=(1,1))
axW=fig.add_subplot(2,2,1)
imgW=axW.matshow(wpic,cmap=cm.gray)
axA=fig.add_subplot(2,2,2)
A=model.M.get_value().reshape((nv*nc,nc)).T
imgA=axA.matshow(A)
axv=fig.add_subplot(2,2,3)
v0=np.zeros((nt,nv)).astype(np.float32)
losshist=[0.0]
#vlines,=axv.plot(losshist)
vlines=axv.plot(v0)

c0=np.dot(data_raw,model.W.get_value().T).astype(np.float32)/100.0
axvC=fig.add_subplot(2,2,4)
imgvC=axvC.matshow(np.eye(nv))



oldloss=9999.0
step=4e-3
nsteps=200
lrate=4e-3

plotting=True
plot_every=20
save_every=100

for i in range(10000):
	oldparams=[]
	oldmoms=[]
	#I store the parameters before doing any learning in case things
	#blow up
	for param in model.params:
		oldparams.append(param.get_value())
	for mom in model.momentums:
		oldmoms.append(mom.get_value()*0.0)
	
	ch,vh,resid_var,res=learn(lrate,step,nsteps,c0*0.0,v0*0.0)
	
	
	#Checks for unstable optimization and resets parameters to last
	#known "good" state if needed. Also changes MAP optimization 
	#hyperparameters to promote stability.
	if res[-1]>res[0] or np.isnan(res[-1]):
		for i in range(len(model.params)):
			model.params[i].set_value(oldparams[i])
			model.momentums[i].set_value(oldmoms[i]*0.99)
		step=step*0.9
		
		print 'Instability detected'
		#objlines.set_ydata(res)
		#objlines.set_xdata(np.arange(res.shape[0]))
		#axobj.relim()
		#axobj.autoscale_view(True,True,True)
		#vlines.set_ydata(losshist)
		#vlines.set_xdata(np.arange(len(losshist)))
		#axv.relim()
		#axv.autoscale_view(True,True,True)
		#fig.canvas.draw()
		continue
	
	else:
		losshist.append(res[-1])
		oldloss=res[-1]
		c0=ch
		v0=vh
		scaled_res=(res-np.min(res))/(np.max(res)-np.min(res))
		popt, pcov = sio.curve_fit(lossfunc,np.arange(nsteps)/float(nsteps),scaled_res)
		step=step*1.005
		if popt[0]<2.5:
			if nsteps<400:
				nsteps+=5
			
			#objlines.set_ydata(res)
			#objlines.set_xdata(np.arange(res.shape[0]))
			#axobj.relim()
			#axobj.autoscale_view(True,True,True)
			#vlines.set_ydata(losshist)
			#vlines.set_xdata(np.arange(len(losshist)))
			#axv.relim()
			#axv.autoscale_view(True,True,True)
			#fig.canvas.draw()
		
	
	#print np.exp(model.ln_v_var.get_value())
	#print np.mean(np.exp(model.ln_c_var.get_value()))
	print 'Objective: ', res[-1]
	print 'Residual variance: ', resid_var
	print 'M norm: ', np.sqrt(np.sum(model.M.get_value()**2))
	print 'Stepsize: ', step
	print 'nsteps: ', nsteps
	
	if i%plot_every==0 and plotting:
		#pp.plot(res)
		#pp.figure(2)
		#pp.plot(res[:-1]/res[1:],'r')
		#pp.plot((res[:-1]-res[1:])/(res[:-1]),'b')
		#pp.figure(3)
		#pp.plot(ch)
		#pp.figure(4)
		#pp.plot(vh,'r')
		W=model.W.get_value()
		wpic=tile_raster_images(W,(int(np.sqrt(nx)),int(np.sqrt(nx))),(int(np.sqrt(nc)),int(np.sqrt(nc))),tile_spacing=(1,1))
		imgW.set_data(wpic)
		A=model.M.get_value().reshape((nv*nc,nc)).T
		imgA.set_data(A)
		for i in range(nv):
			vlines[i].set_ydata(vh[:,i])
		axv.relim()
		axv.autoscale_view(True,True,True)
		imgW.autoscale()
		imgA.autoscale()
		vC=np.dot(vh.T,vh)/float(vh.shape[0])
		imgvC.set_data(vC)
		imgvC.autoscale()
		fig.canvas.draw()
	
	if i%save_every==0:
		f=open('params_iter_'+i+'.cpl','wb')
		cp.dump(model.params,f,2)
		f.close()


f=open('end_params.cpl','wb')
cp.dump(model.params,f,2)
f.close()
