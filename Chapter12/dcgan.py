import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.datasets import fashion_mnist


# Set random seed for reproducibility
np.random.seed(1000)
tf.set_random_seed(1000)


nb_samples = 5000
code_length = 100
nb_epochs = 200
batch_size = 128
nb_iterations = int(nb_samples / batch_size)


def generator(z, is_training=True):
    with tf.variable_scope('generator'):
        conv_0 = tf.layers.conv2d_transpose(inputs=z,
                                            filters=1024,
                                            kernel_size=(4, 4),
                                            padding='valid')

        b_conv_0 = tf.layers.batch_normalization(inputs=conv_0, training=is_training)

        conv_1 = tf.layers.conv2d_transpose(inputs=tf.nn.leaky_relu(b_conv_0),
                                            filters=512,
                                            kernel_size=(4, 4),
                                            strides=(2, 2),
                                            padding='same')

        b_conv_1 = tf.layers.batch_normalization(inputs=conv_1, training=is_training)

        conv_2 = tf.layers.conv2d_transpose(inputs=tf.nn.leaky_relu(b_conv_1),
                                            filters=256,
                                            kernel_size=(4, 4),
                                            strides=(2, 2),
                                            padding='same')

        b_conv_2 = tf.layers.batch_normalization(inputs=conv_2, training=is_training)

        conv_3 = tf.layers.conv2d_transpose(inputs=tf.nn.leaky_relu(b_conv_2),
                                            filters=128,
                                            kernel_size=(4, 4),
                                            strides=(2, 2),
                                            padding='same')

        b_conv_3 = tf.layers.batch_normalization(inputs=conv_3, training=is_training)

        conv_4 = tf.layers.conv2d_transpose(inputs=tf.nn.leaky_relu(b_conv_3),
                                            filters=1,
                                            kernel_size=(4, 4),
                                            strides=(2, 2),
                                            padding='same')

        return tf.nn.tanh(conv_4)


def discriminator(x, is_training=True, reuse_variables=True):
    with tf.variable_scope('discriminator', reuse=reuse_variables):
        conv_0 = tf.layers.conv2d(inputs=x,
                                  filters=128,
                                  kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding='same')

        conv_1 = tf.layers.conv2d(inputs=tf.nn.leaky_relu(conv_0),
                                  filters=256,
                                  kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding='same')

        b_conv_1 = tf.layers.batch_normalization(inputs=conv_1, training=is_training)

        conv_2 = tf.layers.conv2d(inputs=tf.nn.leaky_relu(b_conv_1),
                                  filters=512,
                                  kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding='same')

        b_conv_2 = tf.layers.batch_normalization(inputs=conv_2, training=is_training)

        conv_3 = tf.layers.conv2d(inputs=tf.nn.leaky_relu(b_conv_2),
                                  filters=1024,
                                  kernel_size=(4, 4),
                                  strides=(2, 2),
                                  padding='same')

        b_conv_3 = tf.layers.batch_normalization(inputs=conv_3, training=is_training)

        conv_4 = tf.layers.conv2d(inputs=tf.nn.leaky_relu(b_conv_3),
                                  filters=1,
                                  kernel_size=(4, 4),
                                  padding='valid')

        return conv_4


if __name__ == '__main__':
    # Load the dataset
    (X_train, _), (_, _) = fashion_mnist.load_data()
    X_train = X_train.astype(np.float32)[0:nb_samples] / 255.0
    X_train = (2.0 * X_train) - 1.0

    width = X_train.shape[1]
    height = X_train.shape[2]

    # Create the graph
    graph = tf.Graph()

    with graph.as_default():
        input_x = tf.placeholder(tf.float32, shape=(None, width, height, 1))
        input_z = tf.placeholder(tf.float32, shape=(None, code_length))
        is_training = tf.placeholder(tf.bool)

        gen = generator(z=tf.reshape(input_z, (-1, 1, 1, code_length)), is_training=is_training)

        r_input_x = tf.image.resize_images(images=input_x, size=(64, 64))

        discr_1_l = discriminator(x=r_input_x, is_training=is_training, reuse_variables=False)
        discr_2_l = discriminator(x=gen, is_training=is_training, reuse_variables=True)

        loss_d_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discr_1_l), logits=discr_1_l))
        loss_d_2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(discr_2_l), logits=discr_2_l))
        loss_d = loss_d_1 + loss_d_2

        loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discr_2_l), logits=discr_2_l))

        variables_g = [variable for variable in tf.trainable_variables() if variable.name.startswith('generator')]
        variables_d = [variable for variable in tf.trainable_variables() if variable.name.startswith('discriminator')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            training_step_d = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss=loss_d, var_list=variables_d)
            training_step_g = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss=loss_g, var_list=variables_g)

    # Train the model
    session = tf.InteractiveSession(graph=graph)
    tf.global_variables_initializer().run()

    samples_range = np.arange(nb_samples)

    for e in range(nb_epochs * 5):
        d_losses = []
        g_losses = []

        for i in range(nb_iterations):
            Xi = np.random.choice(samples_range, size=batch_size)
            X = np.expand_dims(X_train[Xi], axis=3)
            Z = np.random.uniform(-1.0, 1.0, size=(batch_size, code_length)).astype(np.float32)

            _, d_loss = session.run([training_step_d, loss_d],
                                    feed_dict={
                                        input_x: X,
                                        input_z: Z,
                                        is_training: True
                                    })
            d_losses.append(d_loss)

            Z = np.random.uniform(-1.0, 1.0, size=(batch_size, code_length)).astype(np.float32)

            _, g_loss = session.run([training_step_g, loss_g],
                                    feed_dict={
                                        input_x: X,
                                        input_z: Z,
                                        is_training: True
                                    })

            g_losses.append(g_loss)

        print('Epoch {}) Avg. discriminator loss: {} - Avg. generator loss: {}'.format(e + 1, np.mean(d_losses),
                                                                                       np.mean(g_losses)))

    # Show some results
    Z = np.random.uniform(-1.0, 1.0, size=(50, code_length)).astype(np.float32)

    Ys = session.run([gen],
                     feed_dict={
                         input_z: Z,
                         is_training: False
                     })

    Ys = np.squeeze((Ys[0] + 1.0) * 0.5 * 255.0).astype(np.uint8)

    fig, ax = plt.subplots(5, 10, figsize=(15, 8))

    for i in range(5):
        for j in range(10):
            ax[i, j].imshow(Ys[(i * 10) + j], cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.show()

    session.close()

