import random
import string

import matplotlib.pyplot as plt
import numpy
from captcha.image import ImageCaptcha

characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)


def gen(batch_size=32):
    X = numpy.zeros((batch_size, height, width, 3), dtype=numpy.uint8)
    y = [numpy.zeros((batch_size, n_class), dtype=numpy.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y):
    y = numpy.argmax(numpy.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


X, y = next(gen(1))
plt.imshow(X[0])
plt.title(decode(y))

from keras.models import *
from keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = convolutional.Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
    x = convolutional.Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)
    x = pooling.MaxPooling2D((2, 2))(x)

x = core.Flatten()(x)
x = core.Dropout(0.25)(x)
x = [core.Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

from keras.utils import plot_model
from IPython.display import Image

plot_model(model, to_file="model.png", show_shapes=True)
Image('model.png')

model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5,
                    validation_data=gen(), nb_val_samples=1280)

X, y = next(gen(1))
y_pred = model.predict(X)
plt.title('real: %s\npred:%s' % (decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')
plt.axis('off')

from tqdm import tqdm


def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = generator.next()
        y_pred = model.predict(X)
        batch_acc += numpy.mean(map(numpy.array_equal, numpy.argmax(y, axis=2).T, numpy.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num


evaluate(model)

model.save('cnn.h5')
