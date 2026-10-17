
import numpy as np
import joblib

from PIL import Image
from skimage.feature import hog

image_path = input("Enter the image path: ")

image = Image.open(image_path).convert("RGB")
image = image.resize((64, 64))

image_array = np.array(image)

features = hog(
    image_array,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    channel_axis=-1
)

features = features.reshape(1, -1)

model = joblib.load("model/svm_dog_cat_model.pkl")

prediction = model.predict(features)[0]

if prediction == 0:
    print("Prediction: Cat")
else:
    print("Prediction: Dog")
