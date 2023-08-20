import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load MNIST data
mnist = tf.keras.datasets.mnist

# split data into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# x_train = x_train / 255
# x_test = x_test / 255

# create model with several layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # 28 x 28 pixels
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # rectified linear unit (relu)
# model.add(tf.keras.layers.Dropout(0.5))  # Drop 50% of the nodes
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dropout(0.5))  # Drop 50% of the nodes
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # output layer - each unit represents a digit
# note: softmax is a probability distribution function, with all digit probabilities adding up to 1

# compile model
# optimizer - how model is updated based on data and loss function
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=5)

# save model
model.save('mnist.model')

# can load the model sicne we've saved it after training it
model = tf.keras.models.load_model('mnist.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

image_number = 1
while os.path.isfile(f"digits/{image_number}.png"):
    img = cv2.imread(f"digits/{image_number}.png")[:,:,0] # read image as grayscale
    img = np.invert(np.array([img])) # invert image to match training data
    # img = tf.keras.utils.normalize(img, axis=1) # normalize image
    prediction = model.predict(img)
    print(f"The number is probably a {np.argmax(prediction)}!") # argmax gives neuron with highest activation
    plt.imshow(img[0], cmap=plt.cm.binary) # show image
    plt.show()
    image_number += 1


