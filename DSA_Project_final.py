from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import cv2
from PIL import Image


 
image = io.imread('tiger.png')
io.imshow(image)
io.show()
 
rows = image.shape[0]
cols = image.shape[1]
  
image = image.reshape(image.shape[0]*image.shape[1],3)
kmeans = KMeans(n_clusters = 16, n_init=10, max_iter=200)
kmeans.fit(image)
 
clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8) //here we are assigning labels to the clusters
labels = np.asarray(kmeans.labels_,dtype=np.uint8 )  
labels = labels.reshape(rows,cols); 
 
np.save('codebook_tiger.npy',clusters)   //here the image is being saved 
io.imsave('compressed_tiger.png',labels); 

centers = np.load('codebook_tiger.npy')
c_image = io.imread('compressed_tiger.png')
 
image = np.zeros((c_image.shape[0],c_image.shape[1],3),dtype=np.uint8 )
for i in range(c_image.shape[0]):                                         //this is the main algorithm
    for j in range(c_image.shape[1]):
            image[i,j,:] = centers[c_image[i,j],:]
io.imsave('reconstructed_tiger.png',image);
io.imshow(image)
io.show()

img1 = Image.open("tiger.png")  //this part helps in saving and opening the image
img2 = cv2.imread("tiger.png")
img3 = Image.open("reconstructed_tiger.png")
img4 = cv2.imread("reconstructed_tiger.png")

cv2.imshow('tiger', img2)
cv2.imshow("reconstructed_tiger.png", img4)

cv2.waitKey()
