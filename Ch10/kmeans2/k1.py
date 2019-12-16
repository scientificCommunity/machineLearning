# -*- coding: utf-8 -*-
from PCV.tools import imtools, pca
from PIL import Image, ImageDraw
from PCV.localdescriptors import sift
from pylab import *
import glob
from scipy.cluster.vq import *

# download_path = "panoimages"  # set this to the path where you downloaded the panoramio images
# path = "/FULLPATH/panoimages/"  # path to save thumbnails (pydot needs the full system path)

download_path = "/data/captcha/images"  # set this to the path where you downloaded the panoramio images
path = "~/mywork/data/k1/"  # path to save thumbnails (pydot needs the full system path)

# list of downloaded filenames
imlist = imtools.get_imlist('/data/captcha/images/')
nbr_images = len(imlist)

# extract features
# featlist = [imname[:-3] + 'sift' for imname in imlist]
# for i, imname in enumerate(imlist):
#    sift.process_image(imname, featlist[i])

featlist = glob.glob('../data/panoimages/*.sift')

matchscores = zeros((nbr_images, nbr_images))

for i in range(nbr_images):
    for j in range(i, nbr_images):  # only compute upper triangle
        print 'comparing ', imlist[i], imlist[j]
        l1, d1 = sift.read_features_from_file(featlist[i])
        l2, d2 = sift.read_features_from_file(featlist[j])
        matches = sift.match_twosided(d1, d2)
        nbr_matches = sum(matches > 0)
        print 'number of matches = ', nbr_matches
        matchscores[i, j] = nbr_matches
print "The match scores is: \n", matchscores

# copy values
for i in range(nbr_images):
    for j in range(i + 1, nbr_images):  # no need to copy diagonal
        matchscores[j, i] = matchscores[i, j]

n = len(imlist)
# load the similarity matrix and reformat
S = matchscores
S = 1 / (S + 1e-6)
# create Laplacian matrix
rowsum = sum(S, axis=0)
D = diag(1 / sqrt(rowsum))
I = identity(n)
L = I - dot(D, dot(S, D))
# compute eigenvectors of L
U, sigma, V = linalg.svd(L)
k = 2
# create feature vector from k first eigenvectors
# by stacking eigenvectors as columns
features = array(V[:k]).T
# k-means
features = whiten(features)
centroids, distortion = kmeans(features, k)
code, distance = vq(features, centroids)
# plot clusters
for c in range(k):
    ind = where(code == c)[0]
    figure()
    gray()
    for i in range(minimum(len(ind), 39)):
        im = Image.open(imlist[ind[i]])
        subplot(5, 4, i + 1)
        imshow(array(im))
        axis('equal')
        axis('off')
show()