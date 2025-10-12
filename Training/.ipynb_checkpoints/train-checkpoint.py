import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

from Configs.config import hyperparams
from Data.MNIST_Loader import load_and_preprocess_data
from Models.mnist_model import build_model, evaluate_model

def train_model(model, ds_train, ds_val, epochs=hyperparams['epochs']):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = "Models/best_model.h5"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[tensorboard_callback, model_checkpoint_callback]
    )
    return history

if __name__ == "__main__":
    # Load data splits
    ds_train, ds_val, ds_test = load_and_preprocess_data()

    # Build model
    model = build_model()

    # Train model with validation, logging, and best-model checkpointing
    history = train_model(model, ds_train, ds_val)

    # Plot training & validation loss and accuracy values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Display hyperparameters on the loss plot
    hp_text = '\n'.join([
        f'Batch size: {hyperparams["batch_size"]}',
        f'Epochs: {hyperparams["epochs"]}',
        f'Learning rate: {hyperparams["learning_rate"]}',
        f'Hidden units 1: {hyperparams["hidden_units_1"]}',
        f'Hidden units 2: {hyperparams["hidden_units_2"]}',
        f'Dropout rate: {hyperparams["dropout_rate"]}',
        f'L2 lambda: {hyperparams["l2_lambda"]}',
        f'Activation 1: {hyperparams["activation_1"]}',
        f'Activation 2: {hyperparams["activation_2"]}'
    ])
    plt.gca().text(
        0.98, 0.02, hp_text,
        fontsize=9,
        ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.7),
        transform=plt.gca().transAxes
    )

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # Load best saved model for evaluation
    best_model = tf.keras.models.load_model("Models/best_model.h5")

    # Evaluate on test set
    test_loss, test_acc = evaluate_model(best_model, ds_test)
    print(f"Final test accuracy: {test_acc:.4f}")
