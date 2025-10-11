import tensorflow as tf
from tensorflow.keras import layers, models
from Configs.config import hyperparams

def build_model(hidden_units=hyperparams['hidden_units'], dropout_rate=hyperparams['dropout_rate'], learning_rate=hyperparams['learning_rate']):
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(hidden_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_model(model, ds_test):
    test_loss, test_acc = model.evaluate(ds_test)
    print(f'Test accuracy: {test_acc:.4f}')
    return test_loss, test_acc

if __name__ == "__main__":
    # Example usage for quick sanity check (will require the data module)
    from data.MNIST_Loader import load_and_preprocess_data
    ds_train, ds_val, ds_test = load_and_preprocess_data()
    model = build_model()
    model.fit(ds_train, validation_data=ds_val, epochs=hyperparams['epochs'])
    evaluate_model(model, ds_test)
