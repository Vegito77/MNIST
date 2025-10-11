import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from Configs.config import hyperparams
from Data.MNIST_Loader import load_and_preprocess_data
from Models.mnist_model import build_model, evaluate_model

def train_model(model, ds_train, ds_val, epochs=hyperparams['epochs']):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[tensorboard_callback]
    )
    return history

if __name__ == "__main__":
    # Load data splits
    ds_train, ds_val, ds_test = load_and_preprocess_data()

    # Build model
    model = build_model()

    # Train model with validation and logging
    history = train_model(model, ds_train, ds_val)

    # Evaluate on test set
    test_loss, test_acc = evaluate_model(model, ds_test)
    print(f"Final test accuracy: {test_acc:.4f}")
