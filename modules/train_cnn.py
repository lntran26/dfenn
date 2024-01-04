import os

os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "3"  # this is to silence some of TF warning messages

import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks
from keras.models import Sequential
import keras_tuner as kt


# todo: add tuning, save output to keras default rather than pickle


def create_dfe_cnn(input_shape: tuple, n_outputs: int):
    model = Sequential()
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=5,
            strides=2,
            input_shape=input_shape,
            activation="relu",
        )
    )
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.Conv1D(filters=16, kernel_size=2, strides=2, activation="relu"))
    model.add(layers.AveragePooling2D(pool_size=(20, 1)))
    model.add(layers.AveragePooling2D(pool_size=(1, 4)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(n_outputs, activation="relu"))
    model.compile(optimizer="adam", loss="mse", metrics=["mean_squared_error"])

    return model, {}


def model_builder(hp):
    input_shape = (20, 300, 2)
    model = Sequential()
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=5,
            strides=2,
            input_shape=input_shape,
            activation="relu",
        )
    )
    # A hyperparamter for whether to use dropout layer.
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.Conv1D(filters=16, kernel_size=2, strides=2, activation="relu"))
    model.add(layers.AveragePooling2D(pool_size=(20, 1)))
    model.add(layers.AveragePooling2D(pool_size=(1, 4)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(2, activation="relu"))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    # hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e2, step=10, sampling="log"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="mse",
        metrics=["mean_squared_error"],
    )

    return model


def run_tuning(model_builder, train_in, train_out):
    # instantiate the Hyperband tuner
    tuner = kt.Hyperband(model_builder, objective="val_loss", max_epochs=20)

    # Create a callback to stop training early after reaching a certain value for the val_loss
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    # Run the hyperparameter search
    tuner.search(
        train_in, train_out, epochs=30, validation_split=0.3, callbacks=[stop_early]
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(
        f"The hyperparameter search is complete.\
        The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.\
            Dropout: {best_hps.get('dropout')}"
    )

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_in, train_out, epochs=30, validation_split=0.3)

    # Find the optimal number of epochs to train the model
    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
    print("Best epoch: %d" % (best_epoch,))

    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner.hypermodel.build(best_hps)
    # Retrain the model
    hypermodel.fit(train_in, train_out, epochs=best_epoch, validation_split=0.3)

    return hypermodel


def create_dfe_cnn_afs(input_shape: tuple, n_outputs: int):
    model = Sequential()
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=1,
            input_shape=input_shape,
            activation="relu",
        )
    )
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(filters=16, kernel_size=2, strides=2, activation="relu"))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(8, activation="relu"))
    model.add(layers.Dense(n_outputs, activation="relu"))
    model.compile(optimizer="adam", loss="mse", metrics=["mean_squared_error"])

    return model, {}
