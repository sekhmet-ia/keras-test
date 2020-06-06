# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow version: ' + tf.__version__)

# Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

# The train_images and train_labels arrays are the training set that the model uses to learn.
# The model is tested against the test set, test_images and test_labels arrays.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
# Each image is mapped to a single label (from 0 to 9). 
# The class names are not included in the dataset.
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 
    'Sneaker', 'Bag', 'Ankle boot'
]

# Amount of images in the training and test set, and how are they represented.
print('Training data shape: ' + str(train_images.shape)) # (60000, 28, 28)
print('Test data shape: ' + str(test_images.shape)) # (10000, 28, 28)

# Amount of labels in the training and test set.
print('Training labels: ' + str(len(train_labels))) # 60000
print('Test labels: ' + str(len(test_labels))) # 10000

# Preprocess the data
# Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
# Divide the values by 255.
# It's important that the training set and the testing set be preprocessed in the same way.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Check that the data is in the correct format and that you're ready to build and train the network
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Building the neural network requires configuring the layers of the model, then compiling the model.
# Set up the layers
# The tf.keras.layers.Flatten layer transforms the format of the images
# from a two-dimensional array to a one-dimensional array.
# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
# The first Dense layer has 128 nodes or neurons.
# The second layer returns a logits array with length of 10.
# Each node contains a score that indicates the current image belongs to one of the 10 classes.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compile the model.
# Before the model is ready for training, it needs a few more settings.
# Loss function measures how accurate the model is during training.
# Optimizer is how the model is updated based on the data it sees and its loss function.
# Metrics is used to monitor the training and testing steps.
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model.
# Call the model.fit method because it "fits" the model to the training data.
model.fit(train_images, train_labels, epochs=10)
# As the model trains, the loss and accuracy metrics are displayed.

# Evaluate accuracy.
# Compare how the model performs on the test dataset.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

