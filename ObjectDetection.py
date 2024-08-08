import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image

# 1. Load and Prepare the Data
def load_and_prepare_data():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the images to the range [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Convert labels to one-hot encoding

    return (train_images, train_labels), (test_images, test_labels)

# 2. Build the Model
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
   
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Train the Model
def train_model(model, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_split=0.2)

# 4. Evaluate the Model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc:.4f}')

# 5. Predict a Digit
def predict_digit(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    (train_images, train_labels), (test_images, test_labels) = load_and_prepare_data()

    # Build, train, and evaluate the model
    model = build_model()
    train_model(model, train_images, train_labels)
    evaluate_model(model, test_images, test_labels)

    # Predict a digit from a test image
    test_image_path = '/mnt/C/Users/sairam/Downloads/object.jpg'  # Replace with your image path
    predicted_digit = predict_digit(model, test_image_path)
    print(f'Predicted digit: {predicted_digit}')

