import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from tensorflow.keras import mixed_precision

from Configs.config_CNN import hyperparams
from Data.MNIST_Loader import load_and_preprocess_data
from Models.mnist_model_CNN import build_model_CNN, build_ensemble_CNN, evaluate_model_CNN

# Enable mixed precision globally
mixed_precision.set_global_policy('mixed_float16')

def prepare_datasets_CNN():
    ds_train, ds_val, ds_test = load_and_preprocess_data()
    ds_train = ds_train.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    ds_val = ds_val.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    ds_test = ds_test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds_train, ds_val, ds_test

def train_model_CNN(model, ds_train, ds_val, epochs=hyperparams['epochs']):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = "Models/best_model_CNN.h5"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=hyperparams['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping_callback]
    )
    return history

def train_ensemble_CNN(ds_train, ds_val):
    ensemble_models = build_ensemble_CNN()
    histories = []

    for idx, model in enumerate(ensemble_models):
        print(f"\nTraining model {idx+1}/{hyperparams['ensemble_size']}")
        history = train_model_CNN(model, ds_train, ds_val)
        model.save(f"Models/best_model_CNN_{idx+1}.h5")
        histories.append(history)

    return ensemble_models, histories

def evaluate_ensemble_CNN(ensemble_models, ds_test):
    all_preds = []
    for model in ensemble_models:
        preds = model.predict(ds_test)
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)
    y_true = np.concatenate([y.numpy() for _, y in ds_test], axis=0)
    y_pred = np.argmax(avg_preds, axis=1)

    accuracy = np.mean(y_pred == y_true)
    print(f"Ensemble test accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    ds_train, ds_val, ds_test = prepare_datasets_CNN()

    if hyperparams.get('ensemble_size', 1) > 1:
        ensemble_models, histories = train_ensemble_CNN(ds_train, ds_val)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(histories[0].history['loss'], label='Train Loss')
        plt.plot(histories[0].history['val_loss'], label='Validation Loss')
        plt.title('Loss per Epoch (Model 1)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(histories[0].history['accuracy'], label='Train Accuracy')
        plt.plot(histories[0].history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy per Epoch (Model 1)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        test_accuracy = evaluate_ensemble_CNN(ensemble_models, ds_test)
        print(f"Final ensemble test accuracy: {test_accuracy:.4f}")

    else:
        model = build_model_CNN()
        history = train_model_CNN(model, ds_train, ds_val)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        hp_text = '\n'.join([
            f'Batch size: {hyperparams["batch_size"]}',
            f'Epochs: {hyperparams["epochs"]}',
            f'Learning rate: {hyperparams["learning_rate"]}',
            f'Dropout rate: {hyperparams["dropout_rate"]}',
            f'L2 lambda: {hyperparams["l2_lambda"]}',
            f'Activation 1: {hyperparams["activation_1"]}',
            f'Early stopping patience: {hyperparams["early_stopping_patience"]}'
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

        best_model = tf.keras.models.load_model("Models/best_model_CNN.h5")
        test_loss, test_acc = evaluate_model_CNN(best_model, ds_test)
        print(f"Final test accuracy: {test_acc:.4f}")
