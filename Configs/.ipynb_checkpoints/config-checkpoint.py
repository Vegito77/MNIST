hyperparams = {
    'batch_size': 32,             # Typical batch size for stable mini-batch training
    'epochs': 50,                 # Slightly longer training for better convergence
    'learning_rate': 0.00005,       # Standard small learning rate for Adam optimizer
    'hidden_units_1': 1000,        # Reasonable size for first hidden layer to capture complexity
    'hidden_units_2': 750,      # Optional second hidden layer if you want deeper model
	'early_stopping_patience': 15,
    'dropout_rate': 0.5,          # Moderate dropout to reduce overfitting
    'l2_lambda': 0.008,           # Small L2 penalty to regularize weights gently
    'input_shape': (28, 28, 1),   # MNIST image shape (grayscale single channel)
    'activation_1': 'relu',        # ReLU preferred for faster, effective training over sigmoid
    'activation_2': 'relu'      # If second layer used, ReLU recommended here too
}
