import os
import cv2
import numpy as np

IMG_SIZE = (128,128)

real_dir = "data/processed/real"
fake_dir = "data/processed/deepfake"

data = []
labels = []

print("Loading REAL images...")

for img in os.listdir(real_dir):
    path = os.path.join(real_dir, img)

    image = cv2.imread(path)
    if image is None:
        continue

    image = cv2.resize(image, IMG_SIZE)

    data.append(image)
    labels.append(0)

print("Loading FAKE images...")

for img in os.listdir(fake_dir):
    path = os.path.join(fake_dir, img)

    image = cv2.imread(path)
    if image is None:
        continue

    image = cv2.resize(image, IMG_SIZE)

    data.append(image)
    labels.append(1)

data = np.array(data, dtype="float32")
labels = np.array(labels)

print("Total samples:", len(data))

# shuffle
indices = np.arange(len(data))
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

np.save("data/processed/augmented_data.npy", data)
np.save("data/processed/augmented_labels.npy", labels)

print("Dataset saved successfully")