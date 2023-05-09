import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os

# Define a filename to save the model
filename = 'house_model.h5'

# GRADED FUNCTION: house_model
def house_model():
    ### START CODE HERE
    
    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    bedroom_per_house = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    house_cost = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
    
    # Define your model (should be a model with 1 dense layer and 1 unit)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    
    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    # Check if file exists before training the model
    if os.path.isfile(filename):
        print('Model already exists. Loading...')
        model = load_model(filename)
    else:
        print('Training model...')
        # Train your model for 1000 epochs by feeding the i/o tensors
        model.fit(bedroom_per_house, house_cost, epochs=1000)
        
        # Save model after being trained
        model.save(filename)
    
    # Prompt the user to input the number of bedrooms for their desired house
    user_bedrooms = float(input('Enter the number of bedrooms: '))
    # Use the trained model to predict the cost of the house with the given number of bedrooms
    predicted_cost = model.predict([user_bedrooms])[0]

    # Scale the predicted cost by a factor of 100 to represent it in hundreds of thousands
    predicted_cost_scaled = predicted_cost * 100

    # Convert the scaled predicted cost to a string and append the unit ('hundreds of thousands')
    predicted_cost_str = str(predicted_cost_scaled.tolist()[0]) + ' hundreds of thousands'

    # Print the predicted cost for the house with the given number of bedrooms
    print(predicted_cost_str)

    ### END CODE HERE
    return model

# Call the function to execute the code
house_model()
