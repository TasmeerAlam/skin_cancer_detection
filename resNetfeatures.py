from sklearn.preprocessing import LabelEncoder
from melanomaTest.io import HDF5DatasetWriter
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from imutils import paths
import numpy as np
import progressbar
import argparse
import os
import random
import csv
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-o", "--output", required=True,help="path to output HDF5 file")
ap.add_argument("-o2", "--output2", required=True,help="path to output path txt file")
ap.add_argument("-b", "--batch-size", type=int, default=16,help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,help="size of feature extraction buffer")
args = vars(ap.parse_args())

# store the batch size in a convenience variable
bs = args["batch_size"]
# open the output file for writing 
output = open(args["output2"], "w")
# grab the list of images that we'll be describing
print("[INFO] loading images...")

imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

dict={}

with open('lesionlist.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	
	for row in reader:
		
		dict [row['imageid']+".png"]=row['category']

label = []
imPaths = []
# loop over the image and mask paths
for imagePath in imagePaths:
	
	if 'mask' in imagePath:  #skip the mask images

		continue
	imPath = imagePath
	labels = imagePath.split(os.path.sep)[-1]
	category = dict[labels]
	label.append(category)
	imPaths.append(imPath)


for imagefilename in imPaths:	
		output.write(imagefilename+'\n')

output.close()

le = LabelEncoder()
labels = le.fit_transform(label)

print(labels)



print("[INFO] loading network...")
model = ResNet50(weights="imagenet",include_top = False)

dataset = HDF5DatasetWriter((len(imPaths), 100352), args["output"], dataKey = "features", bufSize = args["buffer_size"])
dataset.storeClassLabels(le.classes_)

widgets = ["Extracting Features: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", progressbar.ETA()]

pbar = progressbar.ProgressBar(maxval=len(imPaths),widgets=widgets).start()

for i in np.arange(0, len(imPaths), bs):
	batchPaths = imPaths[i:i + bs]
	batchLabels = labels[i:i + bs]
	batchImages = []

	for (j, imagePath) in enumerate(batchPaths):
		image = load_img(imagePath, target_size = (224, 224))
		image  = np.asarray(image)
		#image =  cv2.Laplacian(image,cv2.CV_64F)

		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)

		batchImages.append(image)

	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=bs)
	features = features.reshape((features.shape[0], -1))

	dataset.add(features, batchLabels)
	pbar.update(i)

dataset.close()
pbar.finish()
