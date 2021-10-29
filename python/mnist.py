# source: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb#scrollTo=SfR4MsSDU880

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import tensorflow.keras as keras

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)), # input layer
        tf.keras.layers.Dense(128, activation='relu'), # layer 2
        # tf.keras.layers.Dense(64, activation='sigmoid'), # layer 3
        # tf.keras.layers.Dropout(0.2), # randomly sets to 0, prevents overfitting
        tf.keras.layers.Dense(10) # output layer
    ])

    # loss fn
    # losses.SCC takes a vector of logits and a True index
    # returns a scalar loss for each example
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy']) # todo: what is this?

    return model

model = create_model()
# for each example, model returns a vector of logits (log-odds scores)
# one for each class
# predictions = model(x_train[:1]).numpy()
# predictions

# converts logits to probabilities for each class
# tf.nn.softmax(predictions).numpy()


# loss_fn(y_train[:1], predictions).numpy()



def ascii(value):
    value = round(int(value * 100))
    if value < 5:
        return ' '
    if value < 10:
        return '.'
    elif value < 20:
        return ':'
    elif value < 30:
        return ';'
    elif value < 40:
        return 'i'
    elif value < 50:
        return '/'
    elif value < 60:
        return 'I'
    elif value < 70:
        return 'T'
    elif value < 80:
        return 'Y'
    elif value < 90:
        return 'H'
    elif value <= 100:
        return '#'


while (True):
    print()
    print("1 - Train")
    print("2 - Load model")
    print("3 - Save model")
    print("4 - Test data")
    print("5 - Training data")
    print("6 - Print images")
    print("0 - Exit")

    userIn = int(input("Option: "))
    
    if userIn == 1:
        # approximage the function using the w/b in the model
        epochs = int(input("Epochs to train: "))
        model.fit(x_train, y_train, epochs=epochs)

        model.summary()

    elif userIn == 2:
        # load file
        filename_load = input("Name of file: ")
        filename_load = "save_data\\" + filename_load + ".model"
        model.load_weights(filename_load)
    elif userIn == 3:
        # save file
        filename_save = input("Name your model: " )
        filename_save = "save_data\\" + filename_save + ".model"
        model.save_weights(filename_save)
        print("{} saved!".format(filename_save))
    elif userIn == 4:
        # checks performance on test set
        model.evaluate(x_test, y_test, verbose=2)
    elif userIn == 5:
        model.evaluate(x_train, y_train, verbose = 2)
    elif userIn == 6:
        a = 1
        for image in x_train.tolist():
            if a == 1:
                for line in image:
                    for value in line:
                        value = ascii(value)
                        print(value, end=' ')
                    print()
                a = int(input("to continue enter '1', any other input will exit: "))
            else:
                break
    elif userIn == 0:
        exit(0)
