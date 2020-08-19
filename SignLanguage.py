import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
def get_data(filename):
  
    input_array = np.loadtxt(filename, delimiter = ",", skiprows  =1).astype('float')
    labels = input_array[:,0]
    images = input_array[:, 1:]
    images = np.reshape(images,(-1, 28,28))

    
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)


print(training_images.shape)
print(training_labels)
print(testing_images.shape)
print(testing_labels.shape)

training_images = np.expand_dims(training_images , axis = -1)

testing_images = np.expand_dims(testing_images , axis = -1)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale = 1./255)

train_generator = train_datagen.flow(
    training_images,
    training_labels,
    
    batch_size=128
)

validation_generator = validation_datagen.flow(
    testing_images,
    testing_labels,
   
    batch_size=64
)
    
# Keep These
print(training_images.shape)
print(testing_images.shape)
    

model = tf.keras.models.Sequential([
    # Your Code Here
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')
])

# Compile Model. 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=27455/128,
                              epochs=8,
                              validation_steps=17172/64, 
                              verbose=1)

model.evaluate(testing_images, testing_labels, verbose=0)

#ploting the results works in jupyter notebook because of inline  matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()