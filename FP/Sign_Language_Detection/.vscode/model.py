from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create the model
model = Sequential(name='sequential_3')

# Input layer
model.add(InputLayer(input_shape=(400, 400, 3), name='conv2d_12_input'))

# Conv & Pool Layers
model.add(Conv2D(32, (3, 3), activation='relu', name='conv2d_12'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_12'))

model.add(Conv2D(32, (3, 3), activation='relu', name='conv2d_13'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_13'))

model.add(Conv2D(16, (3, 3), activation='relu', name='conv2d_14'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_14'))

model.add(Conv2D(16, (3, 3), activation='relu', name='conv2d_15'))
model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_15'))

# Flatten and Dense layers
model.add(Flatten(name='flatten_3'))

model.add(Dense(128, activation='relu', name='dense_12'))
model.add(Dropout(0.5, name='dropout_6'))

model.add(Dense(96, activation='relu', name='dense_13'))
model.add(Dropout(0.4, name='dropout_7'))

model.add(Dense(64, activation='relu', name='dense_14'))

# Output layer (8 classes, softmax for classification)
model.add(Dense(8, activation='softmax', name='dense_15'))

# Compile the model
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Show the model structure
model.summary()

print("Started saving.....")

# Save the model
model.save("sign_language_model.h5")

print("Ended Saving....")