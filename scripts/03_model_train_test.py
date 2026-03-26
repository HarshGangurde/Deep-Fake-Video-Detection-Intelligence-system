# Import Required Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from tensorflow.keras.callbacks import EarlyStopping

# Load Data
X = np.load("data/processed/augmented_data.npy")
y = np.load("data/processed/augmented_labels.npy")

print("Dataset shape:", X.shape)

# Normalize Data
X = (X / 255.0).astype(np.float32)

# Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode="nearest"
)

datagen.fit(X_train)

# Load Pretrained VGG16
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(128,128,3)
)

# Freeze backbone
base_model.trainable = False

# Build Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Focal Loss Function
def focal_loss(alpha=0.5, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = alpha * tf.pow((1 - pt), gamma) * bce
        return tf.reduce_mean(loss)
    return loss

# Compile Model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss(alpha=0.5, gamma=2),
    metrics=["accuracy"]
)

model.summary()

# Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train Model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=35,
    callbacks=[early_stopping]
)

# Predict Probabilities
y_pred_proba = model.predict(X_val)

# Find Optimal Threshold
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
best_threshold = thresholds[np.argmax(precisions * recalls)]

print("Optimal Decision Threshold:", best_threshold)

# Apply Threshold
y_pred = (y_pred_proba > best_threshold).astype(int)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save Model
model.save("main_deepfake_detector_v8.h5")

print("\nModel saved successfully!")