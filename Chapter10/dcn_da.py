import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import ReduceLROnPlateau
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, LeakyReLU, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


# Set random seed for reproducibility
np.random.seed(1000)


nb_classes = 10
train_batch_size = 256
test_batch_size = 100
nb_epochs = 100
steps_per_epoch = 1500


if __name__ == '__main__':
    # Load the dataset
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    # Create the augmented data generators
    train_idg = ImageDataGenerator(rescale=1.0 / 255.0,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True,
                                   horizontal_flip=True,
                                   rotation_range=10.0,
                                   shear_range=np.pi / 12.0,
                                   zoom_range=0.25)

    train_dg = train_idg.flow(x=np.expand_dims(X_train, axis=3),
                              y=to_categorical(Y_train, num_classes=nb_classes),
                              batch_size=train_batch_size,
                              shuffle=True,
                              seed=1000)

    test_idg = ImageDataGenerator(rescale=1.0 / 255.0,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True)

    test_dg = train_idg.flow(x=np.expand_dims(X_test, axis=3),
                             y=to_categorical(Y_test, num_classes=nb_classes),
                             shuffle=False,
                             batch_size=test_batch_size,
                             seed=1000)

    # Create the model
    model = Sequential()

    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     input_shape=(X_train.shape[1], X_train.shape[2], 1)))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     padding='same'))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=1e-5),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit_generator(generator=train_dg,
                                  epochs=nb_epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=test_dg,
                                  validation_steps=int(X_test.shape[0] / test_batch_size),
                                  callbacks=[
                                      ReduceLROnPlateau(factor=0.1, patience=1, cooldown=1, min_lr=1e-6)
                                  ])

    # Show the results
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    ax[0].plot(history.history['acc'], label='Training accuracy')
    ax[0].plot(history.history['val_acc'], label='Validation accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(history.history['loss'], label='Training loss')
    ax[1].plot(history.history['val_loss'], label='Validation loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_yticks(np.linspace(0.0, 1.0, 10))
    ax[1].legend()
    ax[1].grid()
    plt.show()

    