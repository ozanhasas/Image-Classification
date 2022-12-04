from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras import layers
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils 
from keras.constraints import maxnorm
import random
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

classes=['Aeroplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

print('Train samples count:', X_train.shape[0])
print('Test samples count:', X_test.shape[0])

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
 
X_train=X_train/255.0 #normalize a image  0...255 to 0....1
X_test=X_test/255.0

Y_train = np_utils.to_categorical(y_train,10) # One-hot encoding example 6 to [0 0 0 0 0 1 0 0 0 0]
Y_test = np_utils.to_categorical(y_test,10)

opt = SGD(lr=0.001, momentum=0.9)

model = Sequential()

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3), kernel_constraint=maxnorm(3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax', kernel_constraint=maxnorm(3)))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=10,batch_size=32)
#model.save('my_model.h5')


model1 = load_model("my_model.h5")
loss,acc=model1.evaluate(X_test,Y_test)

print("Model Accuracy : "+str(acc*100))
print("Model Loss : "+str(loss))

result = model1.predict(X_test)
result_list_index = np.argmax(result, axis=1)

fig, axes = plt.subplots(4, 4, figsize=(10,10))
axes = axes.ravel()
random_int = random.randint(0, 9983)
index = 0
for i in range(random_int, random_int + 16):
    axes[index].imshow(X_test[i])
    axes[index].set_title("True Class:" + str(classes[np.argmax(Y_test[i])]) + " \nPredict Class:" + str(classes[result_list_index[i]])) 
    axes[index].axis('off')
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9, top=0.9, wspace=0.5, hspace=0.4)
    index +=1
    
    

