import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# plit data for training and validation
split_index = int(len(train_images) * 0.8)  # 80% training, 20% validation split
train_images_split, val_images_split = train_images[:split_index], train_images[split_index:]
train_labels_split, val_labels_split = train_labels[:split_index], train_labels[split_index:]

# Data Augmentation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=10,    
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    shear_range=0.1,  
    zoom_range=0.1,   
    horizontal_flip=True  
)

val_datagen = ImageDataGenerator()  

# Prepare generators
train_generator = train_datagen.flow(train_images_split, train_labels_split, batch_size=64)
val_generator = val_datagen.flow(val_images_split, val_labels_split, batch_size=64)

# Build the CNN model with Batch Normalization, Dropout, and additional layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),  
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# save themodel
checkpoint = ModelCheckpoint('mnist_cnn_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Learning rate scheduler
def lr_schedule(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(lr_schedule)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using data generators
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[checkpoint, lr_scheduler, early_stopping]
)

# Save the final model using the .keras format as well
model.save('mnist_final_model.keras')
print("Model saved successfully in .keras format!")

# Function to preprocess a custom image
def preprocess_image(image_path):
    
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    
    
    img_array = img_array / 255.0
    
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict a custom image
def predict_image(image_path):
    
    img_array = preprocess_image(image_path)
    
    
    prediction = model.predict(img_array)
    
    # Display the image and the prediction
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {np.argmax(prediction)}")
    plt.show()
