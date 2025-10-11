import tensorflow as tf
import numpy as np
from data.MNIST_Loader import load_and_preprocess_data

def load_model(model_path):
    """
    Load a saved TensorFlow Keras model from the given path.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def predict_single_image(model, image):
    """
    Predict the class label for a single preprocessed image.
    Image shape expected: (28, 28, 1), pixel values normalized [0,1].
    """
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)  # shape: (1, 28, 28, 1)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return predicted_class, confidence

def batch_predict(model, dataset):
    """
    Run predictions on a batched dataset.
    Returns list of predicted classes and confidence scores.
    """
    predictions_probs = model.predict(dataset)
    predicted_classes = np.argmax(predictions_probs, axis=1)
    confidences = np.max(predictions_probs, axis=1)
    return predicted_classes, confidences

if __name__ == "__main__":
    # Example usage
    
    # Load data
    _, _, ds_test = load_and_preprocess_data()
    
    # Load saved model (update path as needed)
    model_path = "saved_model/mnist_model"
    model = load_model(model_path)
    
    # Predict on batch test set
    predicted_classes, confidences = batch_predict(model, ds_test)
    print(f"Predicted classes for test set: {predicted_classes[:10]}")
    print(f"Confidence scores for first 10: {confidences[:10]}")

    # Predict on a single example from test set
    for image_batch, label_batch in ds_test.take(1):
        image = image_batch[0]  # first image in batch
        true_label = label_batch[0].numpy()
        pred_class, confidence = predict_single_image(model, image)
        print(f"True label: {true_label}, Predicted: {pred_class}, Confidence: {confidence:.4f}")
