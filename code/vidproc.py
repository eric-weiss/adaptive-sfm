import numpy as np
import cPickle as cp
from utils import tile_raster_images
from matplotlib import pyplot as pp
import matplotlib.cm as cm
import cv2

#This script reads video frames, downsamples and whitens them, and
#saves them as a matrix of flattened frames

datadir='../data/'

in_fn=raw_input('Input_filename: ')
out_fn=raw_input('Output filename: ')

downsample_factor=1

vidcap=cv2.VideoCapture(datadir+in_fn)
success, img=vidcap.read()
print img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in range(downsample_factor):
	img=cv2.pyrDown(img)

ylen, xlen = img.shape
print img.shape

xlen=float(xlen)/2.0; ylen=float(ylen)/2.0

X,Y=np.meshgrid(np.arange(-xlen,xlen,1),np.arange(-ylen,ylen,1))

filt=np.sqrt((X/xlen)**2+(Y/ylen)**2)*np.exp(-0.5*(np.sqrt((X/xlen)**2+(Y/ylen)**2)/0.8)**4)


pp.matshow(filt,cmap=cm.gray)
pp.matshow(img,cmap=cm.gray)
pp.show()

frame_counter=1

block=[]
while True:
	success, img=vidcap.read()
	if not success:
		break
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for i in range(downsample_factor):
		img=cv2.pyrDown(img)
	img=np.asarray(img)
	#img=img[:-1,:]
	tr=np.fft.fftshift(np.fft.fft2(img))
	tr=tr*filt
	wht=np.fft.ifft2(np.fft.fftshift(tr))
	block.append(np.asarray(np.real(wht),dtype='float32'))
	
	if frame_counter==1:
		print np.real(wht).dtype
		pp.matshow(np.real(wht),cmap=cm.gray)
		pp.show()
	
	frame_counter+=1
	print frame_counter


print 'Converting to ndarray...'
block=np.asarray(block,dtype='float32')
block=block[:,22:38,22:38]
v=np.mean(np.sum(np.sum(block**2,axis=2),axis=1))
block=block/np.sqrt(v)
print block.shape
nf,xl,yl=block.shape
block=block.reshape((nf,xl*yl))

f=open(datadir+out_fn,'wb')
if len(block)!=0:
	cp.dump(block,f,2)

f.close()

