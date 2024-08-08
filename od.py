from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Load and Prepare the Data
def load_and_prepare_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    return (train_images, train_labels), (test_images, test_labels)

# 2. Build the Model
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image = np.array(image) / 255.0  # Normalize
        image = image.reshape(1, 28, 28)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)
        return predicted_digit
    except FileNotFoundError as e:
        print(f"File not found: {image_path}")
        # Handle the error or exit

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    (train_images, train_labels), (test_images, test_labels) = load_and_prepare_data()

    # Build, train, and evaluate the model
    model = build_model()
    train_model(model, train_images, train_labels)
    evaluate_model(model, test_images, test_labels)

    # Predict a digit from a test image
    test_image_path = '/mnt/c/Users/sairam/Downloads/2.jpg'  # Updated for WSL
    predicted_digit = predict_digit(model, test_image_path)
    print(f'Predicted digit: {predicted_digit}')
