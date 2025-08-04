# train_model.py

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Paths
base_dir = "dataset"
img_size = 128
batch_size = 32

# Image scaling (no validation split needed)
datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_data = datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

val_data = datagen.flow_from_directory(
    os.path.join(base_dir, "valid"),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

test_data = datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 0 = real, 1 = fake
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# Save model
model.save("fake_image_detector.h5")
print("✅ Model saved as fake_image_detector.h5")

# Evaluate on test data
loss, acc = model.evaluate(test_data)
print(f"🧪 Test accuracy: {acc:.2f}")
