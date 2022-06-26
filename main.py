import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomCallback(keras.callbacks.Callback):     # inherits from the keras class
    def on_epoch_end(self, epoch, logs=None):
        #print(logs.keys())                          # has information such as accuracy, loss, etc.
        if logs.get('accuracy') > 0.98:
            self.model.stop_training = True

img_height = 600
img_width = 600
batch_size = 20         # really doesn't matter much

model = keras.Sequential([
    layers.Input((img_height, img_width, 3)),       # using color so last value is 3 (rgb)
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

data_training = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/jerome/projects/pistachio_detective/images/',
    labels='inferred',
    label_mode = 'binary',
    color_mode = 'rgb', 
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 73,
    validation_split=0.1,
    subset = 'training',
)

data_validation = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/jerome/projects/pistachio_detective/images/',
    labels='inferred',
    label_mode = 'binary',
    color_mode = 'rgb', 
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 73,
    validation_split=0.1,
    subset = 'validation',
)

save_callback = keras.callbacks.ModelCheckpoint(
    'checkpoint/',              # where models are saved
    save_weights_only=True,     # saving weights instead of full model
    monitor='accuracy',
    save_best_only=False,       # save every epoch
)

model.compile(
    optimizer = tf.optimizers.Adam(), 
    loss = 'sparse_categorical_crossentropy', 
    metrics=['accuracy'],
)

model.fit(
    data_training,
    epochs = 15,
    callbacks=[save_callback, CustomCallback()],
)
