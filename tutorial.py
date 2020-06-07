# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow version: ' + tf.__version__)

# The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
# Each image is mapped to a single label (from 0 to 9). 
# The class names are not included in the dataset.
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 
    'Sneaker', 'Bag', 'Ankle boot'
]

# Colors
RED = 'red'
BLUE = 'blue'

def main():
    # Import the Fashion MNIST dataset
    fashion_mnist = keras.datasets.fashion_mnist

    # The train_images and train_labels arrays are the training set that the model uses to learn.
    # The model is tested against the test set, test_images and test_labels arrays.
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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

    # Make predictions.
    # With the model trained, you can use it to make predictions about some images.
    predictions = model.predict(test_images)
    print('First prediction:', str(predictions[0]))
    # A prediction is an array of 10 numbers that represent the model's confidence.
    # Highest confidence value
    most_confident_value = np.argmax(predictions[0])
    print('First prediction most confident value:', str(most_confident_value))
    print('First prediction model:', class_names[most_confident_value])

    # Verify predictions
    # With the model trained, you can use it to make predictions about some images.
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = BLUE
    else:
        color = RED

    plt.xlabel("{} {:2.0f}% ({})"
        .format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]),
        color=color
    )

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color(RED)
    thisplot[true_label].set_color(BLUE)

if __name__ == '__main__':
    main()
