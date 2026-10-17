
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = []
labels = []

image_size = (64, 64)

folders = {
    "cats": 0,
    "dogs": 1
}

for folder, label in folders.items():
    folder_path = os.path.join("dataset", folder)

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(image_size)

            image_array = np.array(image)

            features = hog(
                image_array,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                channel_axis=-1
            )

            data.append(features)
            labels.append(label)

        except Exception:
            continue

X = np.array(data)
y = np.array(labels)

print("Total images loaded:", len(X))
print("Cats:", np.sum(y == 0))
print("Dogs:", np.sum(y == 1))

if len(X) == 0 or len(np.unique(y)) < 2:
    raise ValueError("Add valid images to both dataset/cats and dataset/dogs.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", class_weight="balanced")
)

print("Training HOG + SVM model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(model, "model/svm_dog_cat_model.pkl")

matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Cat", "Dog"]
)

display.plot(cmap="Blues")
plt.title("Dogs vs Cats - HOG + SVM")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

print("Model saved in model/svm_dog_cat_model.pkl")
print("Confusion matrix saved in results/confusion_matrix.png")
