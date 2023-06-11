from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D , MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from glob import glob

batchSize = 32
epochs = 200

TRAINING_DIR = "C:/Users/sscan/PycharmProjects/EE475Project/train"

NumOfClasses = len(glob("C:/Users/sscan/PycharmProjects/EE475Project/train/*"))

print(NumOfClasses)

train_datagen = ImageDataGenerator(rescale = 1/255.0,
                                   rotation_range = 30,
                                   zoom_range = 0.4,
                                    horizontal_flip=True,
                                    shear_range = 0.4)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=batchSize,
                                                    class_mode= 'categorical',
                                                    target_size=(190,190))

VALIDATION_DIR = "C:/Users/sscan/PycharmProjects/EE475Project/validation"
val_datagen = ImageDataGenerator(rescale = 1/255.0)
val_generator = train_datagen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size=batchSize,
                                                    class_mode= 'categorical',
                                                    target_size=(190,190))

callBack = EarlyStopping(monitor='val_loss', patience = 5, verbose = 1, mode = 'auto')

bestModelFilename = "C:/Users/sscan/PycharmProjects/EE475Project/chess_best_model.h5"

bestModel = ModelCheckpoint(bestModelFilename, monitor = 'val_accuracy', verbose = 1, save_best_only = True)

#MODEL

model = Sequential([
    Conv2D(32,(3,3), activation = 'relu', input_shape=(None,None,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3), activation = 'relu'),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3), activation = 'relu'),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3), activation = 'relu'),
    MaxPooling2D(2,2),

    Conv2D(256,(3,3), activation = 'relu'),
    MaxPooling2D(2,2),

    GlobalAveragePooling2D(),

    Dense(512, activation='relu'),
    Dense(512, activation='relu'),

    Dense(NumOfClasses, activation = 'softmax')
])

print(model.summary())

model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics=['accuracy'] )

history = model.fit(train_generator,
                    epochs =epochs,
                    verbose = 1,
                    validation_data = val_generator,
                    callbacks = [bestModel])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc , 'r', label = "Train Accuracy")
plt.plot(epochs, val_acc, 'b', label = "Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, loss , 'r', label = "Train Loss")
plt.plot(epochs, val_loss, 'b', label = "Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


