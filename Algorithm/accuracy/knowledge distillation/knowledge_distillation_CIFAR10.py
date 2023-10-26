import copy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np



class Distiller(keras.Model):
    def __init__(self, student, teacher, chosenType):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.type = chosenType

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):


        # our teacher model is a 10-class classification model
        # our student model is a binary classification model
        # we have to crop the softmax of teacher model from 10-dim to 2-dim based on the chosenType
        # the choseType class in teacher model is the positive class for student model
        def crop10to2(teacher_softmax):
            positive = teacher_softmax[:,chosenType]
            return tf.stack([1 - positive, positive], axis=1)

        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            # distillation_loss = (
            #     self.distillation_loss_fn(
            #         crop10to2(tf.nn.softmax(teacher_predictions / self.temperature, axis=1)),
            #         tf.nn.softmax(student_predictions / self.temperature, axis=1),
            #     )
            #     * self.temperature**2
            # )

            # we use a teacher model that outputs softmax
            distillation_loss = (
                self.distillation_loss_fn(
                    crop10to2(teacher_predictions),
                    student_predictions,
                )
                * self.temperature**2
            )

            # we only have distillation loss as we do not have access to the original training set
            # loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            loss = distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"dl_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


#
# dim_x, dim_y, dim_z = 28, 28, 1  # MNIST
dim_x, dim_y, dim_z = 32, 32, 3  # CIFAR10

# Create the teacher
# teacher = keras.Sequential(
#     [
#         keras.Input(shape=(dim_x, dim_y, dim_z)),
#         layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
#         layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
#         layers.Flatten(),
#         layers.Dense(10),
#     ],
#     name="teacher",
# )

# Create the student
# student = keras.Sequential(
#     [
#         keras.Input(shape=(dim_x, dim_y, dim_z)),
#         layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
#         layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
#         layers.Flatten(),
#         layers.Dense(2),
#     ],
#     name="student",
# )

input_shape = (dim_x, dim_y, dim_z)


# teacher = keras.Sequential(
#     [
#         keras.Input(shape=(dim_x, dim_y, dim_z)),
#         layers.Conv2D(16, (3, 3)),
#         layers.LeakyReLU(alpha=0.2),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(32, (3, 3)),
#         layers.LeakyReLU(alpha=0.2),
#         layers.MaxPooling2D((2, 2)),
#         # layers.Conv2D(64, (3, 3)),
#         # layers.LeakyReLU(alpha=0.2),
#         layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         # layers.Dense(512, activation='relu'),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(10),
#     ],
#     name="teacher",
# )

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10))
teacher = model



model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
student = model

# # Create the student
# student = keras.Sequential(
#     [
#         keras.Input(shape=(dim_x, dim_y, dim_z)),
#         layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
#         layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
#         layers.Flatten(),
#         layers.Dense(2),
#     ],
#     name="student",
# )

# Clone student for later comparison
# we create 10 student model, each of which is a binary classification model
students = []
for i in range(10):
    students.append(keras.models.clone_model(student))



# Prepare the train and test dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# if MNIST, we need to reshape
x_train = np.reshape(x_train, (-1, dim_x, dim_y, dim_z))
x_test = np.reshape(x_test, (-1, dim_x, dim_y, dim_z))

# for teacher model, 10-class classification
x_train_teacher = copy.deepcopy(x_train)
x_test_teacher = copy.deepcopy(x_test)

# for student model, binary classification
chosenType = 0
y_test_stu = (y_test == chosenType).astype(int)
y_train_stu = (y_train == chosenType).astype(int)

x_train_stu = copy.deepcopy(x_train)
x_test_stu = copy.deepcopy(x_test)


# balance dataset
# pos_idx = np.where(y_train_stu == 1)[0]
# neg_idx = np.where(y_train_stu == 0)[0][:len(pos_idx)]
# idx = np.concatenate((pos_idx, neg_idx))
# x_train_stu = x_train[idx]
# y_train_stu = y_train_stu[idx]
#
# pos_idx = np.where(y_test_stu == 1)[0]
# neg_idx = np.where(y_test_stu == 0)[0][:len(pos_idx)]
# idx = np.concatenate((pos_idx, neg_idx))
# x_test_stu = x_test[idx]
# y_test_stu = y_test_stu[idx]


# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'],
)

checkpoint_filepath = './pretrained/cifar10/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False)


# Train and evaluate teacher on data.
# teacher.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=model_checkpoint_callback)

teacher = tf.keras.models.load_model('./checkpoint/cifar10/softmax/MobileNet/best')
teacher.evaluate(x_test, y_test)

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher, chosenType=0)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=['acc'],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3,
)

# Distill teacher to student
distiller.fit(x_train_stu, y_train_stu, epochs=20, validation_data=(x_test_stu, y_test_stu))

# Evaluate student on test dataset
# distiller.evaluate(x_test, y_test_stu)


# # Train student as doen usually
# student_scratch.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )
#
# # Train and evaluate student trained from scratch.
# student_scratch.fit(x_train, y_train, epochs=3)
# student_scratch.evaluate(x_test, y_test)