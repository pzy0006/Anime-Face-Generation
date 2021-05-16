# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:11:00 2021

@author: pengfeiyao
"""
import os
import cv2
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Activation
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Conv2D, Conv2DTranspose, Dropout, UpSampling2D, MaxPooling2D,Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical,plot_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import csv
HAIRS = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
EYES = ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes','brown eyes', 'red eyes', 'blue eyes']
print("number of type hairs：{0},number of type eyes:{1}".format(len(HAIRS),len(EYES)))

#loading images and  the correspounding labels
with open('tags_clean.csv','r') as file:
    lines = csv.reader(file, delimiter = ',')
    y_hairs = []
    y_eyes = []
    y_index = []
    for i, line in enumerate(lines):
        #idx is images' name
        idx = line[0]
        #tags is the description of image
        #it may includes hair eyes doll and so on.
        #see more in tas file
        tags = line[1]
        tags = tags.split('\t')[:-1]
        y_hair = []
        y_eye = []
        for tag in tags:
            tag = tag[:tag.index(':')]
            if(tag in HAIRS):
                y_hair.append(HAIRS.index(tag))
            if(tag in EYES):
                y_eye.append(EYES.index(tag))
        #some of descriptions in tags file does not include both hair and eye
        #right here, we will select the useful tags for the correspounding image
        #the useful tag should always incudes hair and eye
        if (len(y_hair) == 1 and len(y_eye) == 1):
            y_hairs.append(y_hair)
            y_eyes.append(y_eye)
            y_index.append(i)
    y_eyes = np.array(y_eyes)
    y_hairs = np.array(y_hairs)
    y_index = np.array(y_index)
    print("the number of useful tags: ", len(y_index))
#now, create dataset for each images
images_data = np.zeros((len(y_index),64,64,3))
print("starting to create image dataset")
for index,file_index in enumerate (y_index):
    
    images_data[index] = cv2.cvtColor(
        cv2.resize(
            cv2.imread(os.path.join("faces", str(file_index) + '.jpg'), cv2.IMREAD_COLOR),
            (64, 64)),cv2.COLOR_BGR2RGB
            )
images_data = (images_data / 127.5) -1
def build_generator_model(noise_dim, hair_num_class, eye_num_class):
    """
    generator
    :param noise_dim: noise 
    :param hair_num_class: number of  hair type
    :param eye_num_class: number of eye type
    :return: generator
    """
    kernel_init = 'glorot_uniform'

    model = Sequential(name='generator')

    model.add(Reshape((1, 1, -1), input_shape=(noise_dim + 16,)))
    model.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(1, 1), padding="valid",
                              data_format="channels_last", kernel_initializer=kernel_init, ))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same",
                              data_format="channels_last", kernel_initializer=kernel_init))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same",
                              data_format="channels_last", kernel_initializer=kernel_init))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same",
                              data_format="channels_last", kernel_initializer=kernel_init))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format="channels_last",
                     kernel_initializer=kernel_init))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding="same",
                              data_format="channels_last", kernel_initializer=kernel_init))
    model.add(Activation('tanh'))

    latent = Input(shape=(noise_dim,))
    eyes_class = Input(shape=(1,), dtype='int32')
    hairs_class = Input(shape=(1,), dtype='int32')

    hairs = Flatten()(Embedding(hair_num_class, 8, init='glorot_normal')(hairs_class))
    eyes = Flatten()(Embedding(eye_num_class, 8, init='glorot_normal')(eyes_class))
    con = Concatenate()([latent, hairs, eyes])
    fake_image = model(con)
    m = Model(input=[latent, hairs_class, eyes_class], output=fake_image)
    return m

G = build_generator_model(100,len(HAIRS),len(EYES))
def build_discriminator_model(hair_num_class, eye_num_class):
    """
    discriminator
    :param hair_num_class: 
    :param eye_num_class: 
    :return: discriminator
    """
    kernel_init = 'glorot_uniform'
    discriminator_model = Sequential(name="discriminator_model")
    discriminator_model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                   data_format="channels_last", kernel_initializer=kernel_init,
                                   input_shape=(64, 64, 3)))
    discriminator_model.add(LeakyReLU(0.2))
    discriminator_model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                   data_format="channels_last", kernel_initializer=kernel_init))
    discriminator_model.add(BatchNormalization(momentum=0.5))
    discriminator_model.add(LeakyReLU(0.2))
    discriminator_model.add(Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                   data_format="channels_last", kernel_initializer=kernel_init))
    discriminator_model.add(BatchNormalization(momentum=0.5))
    discriminator_model.add(LeakyReLU(0.2))
    discriminator_model.add(Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                   data_format="channels_last", kernel_initializer=kernel_init))
    discriminator_model.add(BatchNormalization(momentum=0.5))
    discriminator_model.add(LeakyReLU(0.2))
    discriminator_model.add(Flatten())
    
    dis_input = Input(shape=(64, 64, 3))

    features = discriminator_model(dis_input)
    
    validity = Dense(1, activation="sigmoid")(features)
    
    label_hair = Dense(hair_num_class, activation="softmax")(features)
    
    label_eyes = Dense(eye_num_class, activation="softmax")(features)
    m = Model(dis_input, [validity, label_hair, label_eyes])
    return m
D = build_discriminator_model(len(HAIRS),len(EYES))
def build_CGAN(gen_lr=0.00015, dis_lr=0.0002, noise_size=100):
    """
    
    :param gen_lr: generator learning rate
    :param dis_lr: discriminator learnig rate
    :param noise_size: noise
    :return:
    """
    gen_opt = Adam(lr=gen_lr, beta_1=0.5)
    G.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])

    dis_opt = Adam(lr=dis_lr, beta_1=0.5)
    losses = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']
    D.compile(loss=losses, loss_weights=[1.4, 0.8, 0.8], optimizer=dis_opt, metrics=['accuracy'])
    D.trainable = False

    opt = Adam(lr=gen_lr, beta_1=0.5)
    gen_inp = Input(shape=(noise_size,))
    hairs_inp = Input(shape=(1,), dtype='int32')
    eyes_inp = Input(shape=(1,), dtype='int32')
    GAN_inp = G([gen_inp, hairs_inp, eyes_inp])
    GAN_opt = D(GAN_inp)
    gan = Model(input=[gen_inp, hairs_inp, eyes_inp], output=GAN_opt)
    gan.compile(loss=losses, optimizer=opt, metrics=['accuracy'])
    return gan
gan = build_CGAN()

def gen_noise(batch_size, noise_size=100):
    """
    Gussian noise
    :param batch_size: # of noise
    :param noise_size: noise size
    :return: （batch_size,noise） Guassian nosie
    """
    return np.random.normal(0, 1, size=(batch_size, noise_size))


def generate_images(generator,img_path):
    """
    G-network generation
    :param generator: 
    :return: （64，64，3）
    """
    noise = gen_noise(16, 100)
    hairs = np.zeros(16)
    eyes = np.zeros(16)

    # get hairs and eyes color
    for h in range(len(HAIRS)):
        hairs[h] = h

    for e in range(len(EYES)):
        eyes[e] = e
    # generating image
    fake_data_X = generator.predict([noise, hairs, eyes])
    plt.figure(figsize=(4, 4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = fake_data_X[i, :, :, :]
        fig = plt.imshow(image)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    # saving image
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)


def sample_from_dataset(batch_size, images, hair_tags, eye_tags):
    """
    randomly pick image
    :param batch_size: batch size
    :param images: dataset
    :param hair_tags: hair color tags
    :param eye_tags: eye color tags
    :return:
    """
    choice_indices = np.random.choice(len(images), batch_size)
    sample = images[choice_indices]
    y_hair_label = hair_tags[choice_indices]
    y_eyes_label = eye_tags[choice_indices]
    return sample, y_hair_label, y_eyes_label
def train(epochs, batch_size, noise_size, hair_num_class, eye_num_class):
    """
    
    :param epochs: epochs
    :param batch_size: batch size
    :param noise_size: noise size
    :param hair_num_class: 
    :param eye_num_class: 
    :return:
    """
    gan_loss = []
    real_data_loss =[]
    fake_data_loss = []
    steps = []
    for step in range(0, epochs):
        
        steps.append(step)
        # save data for each 100 stpes
        if (step % 100) == 0:
            step_num = str(step).zfill(6)
            generate_images(G, os.path.join("./generate_img", step_num + "_img.png"))

        #
        #randomly generating image
        sampled_label_hairs = np.random.randint(0, hair_num_class, batch_size).reshape(-1, 1)
        sampled_label_eyes = np.random.randint(0, eye_num_class, batch_size).reshape(-1, 1)
        sampled_label_hairs_cat = to_categorical(sampled_label_hairs, num_classes=hair_num_class)
        sampled_label_eyes_cat = to_categorical(sampled_label_eyes, num_classes=eye_num_class)
        noise = gen_noise(batch_size, noise_size)
        # G network generating
        fake_data_X = G.predict([noise, sampled_label_hairs, sampled_label_eyes])

        # real data
        real_data_X, real_label_hairs, real_label_eyes = sample_from_dataset(
            batch_size, images_data, y_hairs, y_eyes)
        real_label_hairs_cat = to_categorical(real_label_hairs, num_classes=hair_num_class)
        real_label_eyes_cat = to_categorical(real_label_eyes, num_classes=eye_num_class)

        # smooth
        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        fake_data_Y = np.random.random_sample(batch_size) * 0.2

        # train D network
        dis_metrics_real = D.train_on_batch(real_data_X, [real_data_Y, real_label_hairs_cat,
                                                          real_label_eyes_cat])
        dis_metrics_fake = D.train_on_batch(fake_data_X, [fake_data_Y, sampled_label_hairs_cat,
                                                          sampled_label_eyes_cat])


        noise = gen_noise(batch_size, noise_size)
       
        sampled_label_hairs = np.random.randint(0, hair_num_class, batch_size).reshape(-1, 1)
        sampled_label_eyes = np.random.randint(0, eye_num_class, batch_size).reshape(-1, 1)

   
        sampled_label_hairs_cat = to_categorical(sampled_label_hairs, num_classes=hair_num_class)
        sampled_label_eyes_cat = to_categorical(sampled_label_eyes, num_classes=eye_num_class)

        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        # GAN
        GAN_X = [noise, sampled_label_hairs, sampled_label_eyes]
        # GAN
        GAN_Y = [real_data_Y, sampled_label_hairs_cat, sampled_label_eyes_cat]
        # 
        gan_metrics = gan.train_on_batch(GAN_X, GAN_Y)
        gan_loss.append(gan_metrics[0])
        real_data_loss.append(dis_metrics_real[0])
        fake_data_loss.append(dis_metrics_fake[0])
        # saving generator
        if step % 100 == 0:
            print("Step: ", step)
            print("Discriminator: real/fake loss %f, %f" % (dis_metrics_real[0], dis_metrics_fake[0]))
            print("GAN loss: %f" % (gan_metrics[0]))
        if step % 1000 ==  0:
            G.save(os.path.join('./model', str(step) + "_GENERATOR.hdf5"))
        
    #plot loss
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(steps,real_data_loss,'-',label = "Discriminator real loss")
    lns2 = ax.plot(steps, fake_data_loss, '-', label="Discriminator fake loss")
    lns = lns1 +lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.grid()
    ax.set_xlabel("Train steps")
    ax.set_ylabel("Loss")
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lns3 = ax.plot(steps, gan_loss, '-', label="Generator loss")
    lns = lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.grid()
    ax.set_xlabel("Train steps")
    ax.set_ylabel("Loss")
    plt.show()
    
##################start trian####################
print("################################ start train############################")
train(10000,64,100,len(HAIRS),len(EYES))