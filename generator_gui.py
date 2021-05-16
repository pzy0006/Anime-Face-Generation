import tkinter as tk
from tkinter import ttk

import imageio
import numpy as np
from PIL import Image, ImageTk
from keras.layers import BatchNormalization, Activation, Embedding
from keras.layers import Concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model

num_class_hairs = 12
num_class_eyes = 11


def build_generator():
    kernel_init = 'glorot_uniform'
    latent_size = 100
    model = Sequential()
    model.add(Reshape((1, 1, -1), input_shape=(latent_size + 16,)))
    model.add(
        Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(1, 1), padding="valid", data_format="channels_last",
                        kernel_initializer=kernel_init, ))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(
        Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                        kernel_initializer=kernel_init))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(
        Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                        kernel_initializer=kernel_init))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(
        Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                        kernel_initializer=kernel_init))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last",
                     kernel_initializer=kernel_init))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(
        Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding="same", data_format="channels_last",
                        kernel_initializer=kernel_init))
    model.add(Activation('tanh'))
    # 3 inputs
    latent = Input(shape=(latent_size,))
    eyes_class = Input(shape=(1,), dtype='int32')
    hairs_class = Input(shape=(1,), dtype='int32')
    # embedding
    eyes = Flatten()(Embedding(num_class_eyes, 8, init='glorot_normal')(eyes_class))
    hairs = Flatten()(Embedding(num_class_hairs, 8, init='glorot_normal')(hairs_class))
    # h = merge(, mode='concat')
    h = Concatenate()([latent, hairs, eyes])
    fake_image = model(h)
    m = Model(input=[latent, hairs_class, eyes_class], output=fake_image)
    return m


def load_generators():

    g = build_generator()
    g.load_weights('./model/'+str(18900) + '_GENERATOR.hdf5')
    return g


G = load_generators()
win = tk.Tk()
win.title('GUI')
win.geometry('400x200')


def gen_noise(batch_size, latent_size):
    return np.random.normal(0, 1, size=(batch_size, latent_size))


def generate_images(generator, latent_size, hair_color, eyes_color):
    noise = gen_noise(1, latent_size)
    return generator.predict([noise, hair_color, eyes_color])


def create():
    hair_color = np.array(comboxlist1.current()).reshape(1, 1)
    eye_color = np.array(comboxlist2.current()).reshape(1, 1)

    image = generate_images(G, 100, hair_color, eye_color)[0]
    imageio.imwrite('anime.png', image)
    img_open = Image.open('anime.png')
    img = ImageTk.PhotoImage(img_open)
    label.configure(image=img)
    label.image = img


comvalue1 = tk.StringVar()  
comboxlist1 = ttk.Combobox(win, textvariable=comvalue1)
comboxlist1["values"] = (
    'orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair',
    'blue hair', 'black hair', 'brown hair', 'blonde hair')
comboxlist1.current(0)
comboxlist1.pack()

comvalue2 = tk.StringVar()
comboxlist2 = ttk.Combobox(win, textvariable=comvalue2)
comboxlist2["values"] = (
    'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes',
    'brown eyes', 'red eyes', 'blue eyes')
comboxlist2.current(0)
comboxlist2.pack()

label = tk.Label(win)
label.pack()

b = tk.Button(win,
              text='generate!', 
              width=15, height=2,
              command=create)  
b.pack()

win.mainloop()
