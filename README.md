# Projeft: Anime Face Generation by Using DGAN
### Team number: Pengfei Yao, Ruixuan Zhang
Our Project is aiming to generate anime face by using DGAN. 
## Discriminator:
The input of discriminator only has input image as size(64,64,3). However, the output is [the probability of real image, color oof hair, color of eyes].
In this network, we will do this:
    1.Real image, real label -- able to know what is real image
    2.Feke image, fake label -- able to know what is fake image
The following is architecture and code.
```
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
```
## Generator
The input of Generator is color of hair, color of eye and noise in 100 dim. The output is a RGB image in size of 64 by 64.
The following is architecture and code.
```
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
```
## Train
We train it with 10000 epochs, batch size is 64, noise size is 100.

## Dataset
Labels are in clean_tag.csv file. Anime face images in face drictory. [The link to download.](https://drive.google.com/file/d/1WWwMgYz9VhKgYfZB0nb_jPmJE-qlEPh0/view?usp=sharing)
Some of images are deleted, because the correspounding labels in clean_tag file are not available.

## How to use it
Open with GUI, click generate button. You may need to wait for seconds, when you first time to run. 
## see more in our report file.



