import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from Configs.config import hyperparams

def build_model(
    hidden_units_1=hyperparams['hidden_units_1'],
    dropout_rate=hyperparams['dropout_rate'],
    learning_rate=hyperparams['learning_rate'],
    l2_lambda=hyperparams['l2_lambda'],
    input_shape=hyperparams['input_shape'],
    activation_1=hyperparams['activation_1'],
):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(
            hidden_units_1,
            activation=None,
            kernel_regularizer=regularizers.L2(l2_lambda)
        ),
        layers.BatchNormalization(),
        layers.Activation(activation_1),
        layers.Dropout(dropout_rate),
        layers.Dense(
            10,
            activation=None,  # Use logits, no softmax because mixed precision
            kernel_regularizer=regularizers.L2(l2_lambda)
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_ensemble():
    ensemble_models = []
    for _ in range(hyperparams['ensemble_size']):
        model = build_model()
        ensemble_models.append(model)
    return ensemble_models

def evaluate_model(model, ds_test):
    test_loss, test_acc = model.evaluate(ds_test)
    print(f'Test accuracy: {test_acc:.4f}')
    return test_loss, test_acc
