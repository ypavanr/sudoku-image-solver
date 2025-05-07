import keras
from keras.datasets import mnist # imporint the data set
im_row , im_col = 28, 28 # size of image 28*28
(train_img , train_labels),(test_img , test_labels) = mnist.load_data() 
import matplotlib.pyplot as plt
train_img = train_img.reshape(train_img.shape[0],im_row,im_col,1)
test_img = test_img.reshape(test_img.shape[0],im_row,im_col,1)
input_shape = (im_row,im_col,1)
train_img = train_img/255.0
test_img = test_img/255.0
train_labels = keras.utils.to_categorical(train_labels,10) # 10 because we have 10 numbers so we need 10 classes
test_labels = keras.utils.to_categorical(test_labels, 10)
from keras.models import Sequential
from keras.layers  import Conv2D ,MaxPool2D
from keras.layers  import Flatten ,Dense
from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',kernel_initializer='he_uniform', input_shape=(28, 28, 1))) 
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(3,3),activation = 'relu'))
model.add(Conv2D(128,kernel_size =(3,3),activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100,activation='relu',kernel_initializer = 'he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
opt = keras.optimizers.Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_img, train_labels,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(test_img, test_labels))
score = model.evaluate(test_img,test_labels,verbose=0)
model.save("my_model.keras")
