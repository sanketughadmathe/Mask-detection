import numpy as np
import seaborn as sns
sns.set(style= "ticks")
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

data = np.load("data.npy")
target = np.load("target.npy")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(200, (3,3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50, activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2, activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', save_best_only=True, mode='auto')

history = model.fit(X_train, y_train, epochs=20, callbacks=[checkpoint], validation_split=0.2)

loses_df = pd.DataFrame(history.history)

loses_df[["loss", "val_loss"]].plot(figsize=(10,7))
plt.show()

loses_df[["accuracy", "val_accuracy"]].plot(figsize=(10,7))
plt.show()

model.save("model_wo_tr.h5")