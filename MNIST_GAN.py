import tensorflow as tf
import numpy as np
import png
import os
import shutil
import pickle

tf.enable_eager_execution()


folder = "Data/"+os.path.basename(__file__).split(".")[0]
picture_folder = os.path.join(folder, "pics")
gen_weights = os.path.join(folder, "gen.dat")
disc_weights = os.path.join(folder, "disc.dat")


(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train[(y_train == 4)]

real = x_train[:, :, :, None].astype(np.float32)/255

NOISE_SHAPE = (20,)
BATCH_SIZE = 100


def saveImage(image, name, grayscale=False):

    color_format = "RGB"
    if grayscale:
        color_format = "L"

    png.from_array(np.round((image) * 255).astype("uint8"),
                   color_format).save("{}.png".format(name))


gen = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            units=7*7*4, activation=tf.nn.relu,  input_shape=NOISE_SHAPE),
        tf.keras.layers.Reshape((7, 7, 4)),
        tf.keras.layers.Conv2DTranspose(
            kernel_size=3, strides=(2, 2), filters=2, activation=tf.nn.relu, padding="same"),
        tf.keras.layers.Conv2DTranspose(
            kernel_size=3, strides=(2, 2), filters=1, activation=tf.nn.sigmoid, padding="same"),
        tf.keras.layers.Reshape((28, 28, 1)),
    ],
)

disc = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=5, padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=5, padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=2),
    ]
)
try:
    gen.load_weights(gen_weights)
    disc.load_weights(disc_weights)
except:
    print("No files loaded")

ds = tf.data.Dataset.from_tensor_slices(
    real).shuffle(60000).repeat().batch(BATCH_SIZE)


def disc_loss(pic, categories):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc(pic), labels=categories)


def gen_loss(rand):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc(gen(rand)), labels=tf.zeros(shape=BATCH_SIZE, dtype=tf.int32))


who_next_ratio = 0.5
optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)

if __name__ == "__main__":
    if os.path.isdir(picture_folder):
        shutil.rmtree(picture_folder)
    os.makedirs(picture_folder)

    for i, pic in enumerate(ds):
        rand1 = tf.random_normal(shape=(BATCH_SIZE,) + NOISE_SHAPE)
        rand2 = tf.random_normal(shape=(BATCH_SIZE,) + NOISE_SHAPE)

        fake_pic = gen(rand1)

        optimizer.minimize(lambda:
                           disc_loss(pic, tf.zeros(shape=BATCH_SIZE, dtype=tf.int32)) +
                           disc_loss(fake_pic, tf.ones(shape=BATCH_SIZE, dtype=tf.int32)))
        optimizer.minimize(lambda: gen_loss(rand2), var_list=gen.variables)

        a, b, c = (float(tf.reduce_mean(disc_loss(pic, tf.zeros(shape=BATCH_SIZE, dtype=tf.int32)))), float(tf.reduce_mean(disc_loss(
            fake_pic, tf.ones(shape=BATCH_SIZE, dtype=tf.int32)))), float(tf.reduce_mean(gen_loss(rand2))))

        who_next_ratio = b/(b+c)

        print(a, b, c)

        who_next_ratio = 0.5

        if i % 200 == 0:
            print(i)
            saveImage(fake_pic[0], os.path.join(
                picture_folder, "Image_{}".format(i)), grayscale=True)
            gen.save_weights(gen_weights)
            disc.save_weights(disc_weights)
