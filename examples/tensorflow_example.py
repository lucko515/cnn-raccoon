import tensorflow as tf

model = tf.keras.models.Sequential([
        # YOUR CODE HERE
        tf.keras.layers.BatchNormalization(input_shape=(300, 300, 3)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Convolution2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Convolution2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.Convolution2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.Convolution2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

from cnn_raccoon import inspector

inspector(model, engine="keras")
