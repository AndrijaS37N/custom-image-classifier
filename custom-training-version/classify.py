from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2

# TODO -> Work more on this. Woo woo!

# USAGE:
# python classify.py --model arguments.model --labelbin mlb.pickle --image test-images/test-images-*.jpg

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)

# pre-process the image for classification
image = cv2.resize(image, (300, 300))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained CNN and the multi-label binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

# classify the input image and then find the indices two class labels with the largest probabilities
print("[INFO] classifying image...")
probabilities = model.predict(image)[0]
indices = np.argsort(probabilities)[::-1][:2]

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(indices):
    # build the label and draw the label on the image
    label = "{}: {:.2f}%".format(mlb.classes_[j], probabilities[j] * 100)
    cv2.putText(output, label, (10, (i * 30) + 25),
                cv2.QT_FONT_NORMAL, 0.7, (0, 255, 0), 2)

# show the probabilities for each of the individual labels
i = 0
for (label, p) in zip(mlb.classes_, probabilities):
    i = i + 1
    print("{}. {}: {:.2f}%".format(i, label, p * 100))

# output the image
cv2.imshow("Output", output)
cv2.waitKey(0)
