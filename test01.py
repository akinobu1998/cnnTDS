import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import datetime


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print('before x_train shape:', x_train.shape)

# Reshape the array to 4D

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# end Reshape
print('after x_train shape:', x_train.shape)

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, 
      batch_size=100, 
      epochs=10)

# end model

# evaluation
model.evaluate(x_test, y_test)
# end evaluation

# show sample image

for index in range(1000, 2000, 100):
    image_index = index
    
    print("answer", end="")
    print(y_test[image_index], end="")
    print("  predict", end="")
    # plt.imshow(x_train[image_index].reshape(28, 28), cmap='Greys')
    # plt.show()
    pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
    print(pred.argmax())
