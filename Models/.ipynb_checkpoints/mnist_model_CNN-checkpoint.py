import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from Configs.config_CNN import hyperparams

def build_model_CNN(
    dropout_rate=hyperparams['dropout_rate'],
    learning_rate=hyperparams['learning_rate'],
    l2_lambda=hyperparams['l2_lambda'],
    input_shape=hyperparams['input_shape'],
    activation_1=hyperparams['activation_1'],
):
    model = models.Sequential([
        # 1st Conv-Pool block
        layers.Conv2D(20, kernel_size=(5, 5), padding='same', 
                      input_shape=input_shape,
                      kernel_regularizer=regularizers.L2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation(activation_1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),

        # 2nd Conv-Pool block
        layers.Conv2D(40, kernel_size=(5, 5), padding='same',
                      kernel_regularizer=regularizers.L2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation(activation_1),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),

        # Flatten and fully connected layer
        layers.Flatten(),
        layers.Dense(100,
                     kernel_regularizer=regularizers.L2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation(activation_1),
        layers.Dropout(dropout_rate),

        # Output layer with logits (no softmax)
        layers.Dense(10, activation=None,
                     kernel_regularizer=regularizers.L2(l2_lambda))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_ensemble_CNN():
    ensemble_models = []
    for _ in range(hyperparams['ensemble_size']):
        model = build_model_CNN()
        ensemble_models.append(model)
    return ensemble_models

def evaluate_model_CNN(model, ds_test):
    test_loss, test_acc = model.evaluate(ds_test)
    print(f'Test accuracy: {test_acc:.4f}')
    return test_loss, test_acc
