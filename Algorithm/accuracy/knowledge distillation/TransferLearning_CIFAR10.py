import numpy as np
import tensorflow as tf
from tensorflow import keras


print("Tensorflow version " + tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 32, 32, 3))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 32, 32, 3))

'''
Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
Input size is 224 x 224.
'''



def feature_extractor(inputs):

  # efficientnet_v2.EfficientNetV2S - param=22M
  # resnet.ResNet50 - param=22M
  # mobilenet_v2.MobileNetV2 - param=4M
  # vgg19.VGG19 - param=21M
  # vgg16.VGG16 - param=15M
  # xception.Xception - param=23M
  feature_extractor = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),  # (224, 224, 3)
                                                            include_top=False,
                                                            weights='imagenet')(inputs)
  return feature_extractor


'''
Defines final dense layers and subsequent softmax layer for classification.
'''


def classifier(inputs):
  x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(1024, activation="relu")(x)
  x = tf.keras.layers.Dense(512, activation="relu")(x)
  x = tf.keras.layers.Dense(10, activation='softmax')(x)
  return x


'''
Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
Connect the feature extraction and "classifier" layers to build the model.
'''


def final_model(inputs):
  resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)

  resnet_feature_extractor = feature_extractor(resize)
  classification_output = classifier(resnet_feature_extractor)

  return classification_output


'''
Define the model and compile it. 
Use Stochastic Gradient Descent as the optimizer.
Use Sparse Categorical CrossEntropy as the loss function.
'''


def define_compile_model():
  inputs = tf.keras.layers.Input(shape=(32, 32, 3))

  classification_output = final_model(inputs)
  model = tf.keras.Model(inputs=inputs, outputs=classification_output)

  model.compile(optimizer='SGD',
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])

  # model.compile(
  #   optimizer=keras.optimizers.Adam(),
  #   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #   metrics=['acc'],
  # )
  return model

model = define_compile_model()


# model.summary()

epochs = 30
base = 0

# model = tf.keras.models.load_model('./checkpoint/MNIST/MobileNetV2/epoch_{}'.format(base))

# model.evaluate(x_test, y_test)
# exit()

checkpoint_filepath = './checkpoint/cifar10/softmax/MobileNet/best'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

history = model.fit(x_train, y_train, epochs=epochs, validation_data = (x_test, y_test), batch_size=64, callbacks=model_checkpoint_callback)







