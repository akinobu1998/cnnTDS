import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import datetime

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the array to 4D
def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])


# Creating a Sequential Model and adding the layers
model = create_model()

model.compile(optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

batchsize = 10

with open('logbatch' + str(batchsize) + "-" + datetime.datetime.now().strftime("%H%M%S"), 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['tr' + str(batchsize), 'te' + str(batchsize)])
    hist = model.fit(x=x_train,
        y=y_train,
        epochs=50,
        batch_size=batchsize,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback])
    writer.writerows(map(lambda x, y: [x, y], hist.history['accuracy'], hist.history['val_accuracy']))
    # writer.writerows(map(lambda x: [x], hist.history['val_accuracy']))

# end model

# evaluation
# model.evaluate(x_test, y_test)
# end evaluation

# show sample image

# for index in range(1000, 2000, 100):
#     image_index = index
    
#     print("answer", end="")
#     print(y_test[image_index], end="")
#     print("  predict", end="")
#     # plt.imshow(x_train[image_index].reshape(28, 28), cmap='Greys')
#     # plt.show()
#     pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
#     print(pred.argmax())
