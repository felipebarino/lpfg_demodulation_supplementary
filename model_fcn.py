import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from copy import copy
import pickle

K = keras.backend

def scheduler(epoch, lr):
    """
    Learning rate scheduler function.

    Parameters:
    epoch (int): The current epoch number.
    lr (float): The current learning rate.

    Returns:
    float: The updated learning rate.
    """
    if epoch == 2:
        return 0.5*lr
    if epoch < 5:
        return lr
    else:
        return lr * 0.95

def load_synth_data(dataset, noisy=True, range=(1500, 1600)):
    """
    Load synthetic data from a file.

    Parameters:
    dataset (str or dict): Dataset.
    noisy (bool, optional): Whether the data is noisy or not. Defaults to True.
    range (tuple, optional): The range of values to normalize the data. Defaults to (1500, 1600).

    Returns:
    tuple: The loaded data, including input strength, normalized FBG positions, normalized FBG positions, and target values.
    """
    if not (isinstance(dataset, str) or isinstance(dataset, dict)):
        raise ValueError('The dataset must be a path string to a datset file or a dataset dictionary.')
    if isinstance(dataset, str):
        with open(dataset, 'rb') as file:
            dataset = pickle.load(file)
    X = dataset['input_strength']
    X_c = dataset['input_strength_clean']
    fbg_pos = dataset['wl_bragg']
    y = dataset['target']

    if noisy:
        X = np.append(X_c, X, axis=0)
        y = np.append(y, y, axis=0)
        fbg_pos = np.append(fbg_pos, fbg_pos, axis=0)
    else:
        X = X_c

    fbg_norm = (copy(fbg_pos) - min(range))/(max(range) - min(range)) 

    X = X - X.min(axis=1).reshape(-1, 1).repeat(13, axis=1)
    X = X / X.sum(axis=1).reshape(-1, 1).repeat(13, axis=1)

    mask = np.isnan(X).any(axis=1)
    X = X[~mask]
    fbg_pos = fbg_pos[~mask]
    fbg_norm = fbg_norm[~mask]
    y = y[~mask]
    return X, fbg_pos, fbg_norm, y

def load_measured_data(file_name, range=(1500, 1600)):
    """
    Load measured data from a file.

    Parameters:
    file_name (str): The path to the file containing the data.
    range (tuple, optional): The range of values to normalize the data. Defaults to (1500, 1600).

    Returns:
    tuple: The loaded data, including input strength, normalized FBG positions, normalized FBG positions, and target values.
    """
    with open(file_name, 'rb') as file:
        dataset = pickle.load(file)
    X = dataset['input_strength']
    fbg_pos = dataset['wl_bragg']
    y = dataset['target']

    fbg_norm = (copy(fbg_pos) - min(range))/(max(range) - min(range)) 

    X = X - X.min(axis=1).reshape(-1, 1).repeat(13, axis=1)
    X = X / X.sum(axis=1).reshape(-1, 1).repeat(13, axis=1)

    mask = np.isnan(X).any(axis=1)
    X = X[~mask]
    fbg_pos = fbg_pos[~mask]
    fbg_norm = fbg_norm[~mask]
    y = y[~mask]
    return X, fbg_pos, fbg_norm, y

def build_model(hp):
    """
    Function to build the model based on a set of hyperparameters.

    Parameters:
    hp (kerastuner.HyperParameters): The hyperparameters object.

    Returns:
    tensorflow.python.keras.engine.functional.Functional: The built model.
    """   
    inputs = keras.Input(shape=(13, ), name='input')
    fbg_positions = keras.Input(shape=(13, ), name='fbg_positions')
    fbg_norm_positions = keras.Input(shape=(13, ), name='fbg_norm')

    number_features_1 = hp.Int('layer_1_size', 100, 300, step=10)
    number_features_2 = hp.Int('layer_2_size', 100, 300, step=10)

    hidden_activation = hp.Choice('layer_activation', ['relu', 'tanh', 'sigmoid'])
    att_activation = 'softmax'
    drop_rate = hp.Float('dropout', 0.1, 0.3, step=0.1)

    concat = keras.layers.Concatenate()([inputs, fbg_norm_positions])

    extracted_features = keras.layers.Dense(number_features_1, activation=hidden_activation,
                                name='hidden')(concat)

    extracted_features = keras.layers.Dropout(drop_rate)(extracted_features)

    extracted_features = keras.layers.Dense(number_features_2, activation=hidden_activation,
                                name='hidden2')(extracted_features)

    attention_map = keras.layers.Dense(number_features_1, activation=hidden_activation,
                                name='attention_hidden')(concat)

    attention_map = keras.layers.Dropout(drop_rate)(attention_map)

    attention_map = keras.layers.Dense(number_features_2, activation=att_activation,
                                name='attention')(attention_map)

    extracted_features = keras.layers.multiply([extracted_features, attention_map], name='mult')

    filtered_input = keras.layers.Dense(13, activation='sigmoid',
                                        name='input_filter')(extracted_features)

    output = keras.layers.dot([filtered_input, fbg_positions], axes=1, name='output')

    model = keras.Model(inputs=[inputs, fbg_positions, fbg_norm_positions], outputs=output, name='simple_ann')
        
    model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=[keras.metrics.RootMeanSquaredError(),
                        keras.losses.MeanAbsolutePercentageError()],)

    return model


def build_model_dout(hp):
    """
    Function to build the model based on a set of hyperparameters.

    Parameters:
    hp (kerastuner.HyperParameters): The hyperparameters object.

    Returns:
    tensorflow.python.keras.engine.functional.Functional: The built model.
    """   
    inputs = keras.Input(shape=(13, ), name='input')
    fbg_positions = keras.Input(shape=(13, ), name='fbg_positions')
    fbg_norm_positions = keras.Input(shape=(13, ), name='fbg_norm')

    number_features_1 = hp.Int('layer_1_size', 100, 300, step=10)
    number_features_2 = hp.Int('layer_2_size', 100, 300, step=10)

    hidden_activation = hp.Choice('layer_activation', ['relu', 'tanh', 'sigmoid'])
    att_activation = 'softmax'
    drop_rate = hp.Float('dropout', 0.1, 0.3, step=0.1)

    concat = keras.layers.Concatenate()([inputs, fbg_norm_positions])

    extracted_features = keras.layers.Dense(number_features_1, activation=hidden_activation,
                                name='hidden')(concat)

    extracted_features = keras.layers.Dropout(drop_rate)(extracted_features)

    extracted_features = keras.layers.Dense(number_features_2, activation=hidden_activation,
                                name='hidden2')(extracted_features)

    attention_map = keras.layers.Dense(number_features_1, activation=hidden_activation,
                                name='attention_hidden')(concat)

    attention_map = keras.layers.Dropout(drop_rate)(attention_map)

    attention_map = keras.layers.Dense(number_features_2, activation=att_activation,
                                name='attention')(attention_map)

    extracted_features = keras.layers.multiply([extracted_features, attention_map], name='mult')

    filtered_input = keras.layers.Dense(13, activation='softmax',
                                        name='input_filter')(extracted_features)

    output = keras.layers.dot([filtered_input, fbg_positions], axes=1, name='output')

    model = keras.Model(inputs=[inputs, fbg_positions, fbg_norm_positions], outputs=[output, filtered_input], name='fbg_attention')
        
    model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=[keras.metrics.RootMeanSquaredError(),
                        keras.losses.MeanAbsolutePercentageError()],)

    return model


def predict_with_uncertainty(f, x, n_iter=100):
    """
    Predict the output of a model with uncertainty.

    Parameters:
    f (function): The model function.
    x (array-like): The input data to the model.
    n_iter (int, optional): The number of iterations for uncertainty estimation. Defaults to 100.

    Returns:
    tuple: The predicted output and its uncertainty.
    """
    # Set the learning phase to 1 (training mode)
    K.set_learning_phase(1)
    # Initialize an array to store the results of each iteration
    result = np.zeros((n_iter, x[0].shape[0], 1))
    # Run the model function for each iteration and store the results
    for i in range(n_iter):
        result[i] = f(x)
    # Calculate the mean prediction across all iterations
    prediction = result.mean(axis=0)
    # Calculate the uncertainty of the prediction using the t-distribution
    uncertainty = stats.t.ppf(0.975, n_iter - 1) * result.std(axis=0)/np.sqrt(n_iter)
    # Set the learning phase back to 0 (test mode)
    K.set_learning_phase(0)
    # Return the prediction and its uncertainty
    return prediction.flatten(), uncertainty.flatten()

def dropout_ensamble_prediction(f, x, n_iter=100):
    """
    Predict the output of a model with dropout.

    Parameters:
    f (function): The model function.
    x (array-like): The input data to the model.
    n_iter (int, optional): The number of iterations for dropout estimation. Defaults to 100.

    Returns:
    tuple: The predicted output and its uncertainty.
    """
    # Set the learning phase to 1 (training mode)
    K.set_learning_phase(1)
    # Initialize an array to store the results of each iteration
    result = np.zeros((n_iter, x[0].shape[0], 1))
    # Run the model function for each iteration and store the results
    for i in range(n_iter):
        result[i] = f(x)
    return result


def get_layer_outputs(model, inputs, layer_names):
    """
    Retrieve the output of specific layers in a Keras model.

    Parameters:
    model (tensorflow.python.keras.engine.functional.Functional): The Keras functional model.
    inputs (numpy.ndarray): The input data as a numpy array.
    layer_names (list): A list of names of the layers whose outputs are to be retrieved.

    Returns:
    model_output (numpy.ndarray): The output of the entire model.
    intermediate_outputs (list): A list of numpy arrays representing the output of each specified layer.
    """ 

    # Create a list of output tensors for the layers we're interested in
    # This is done by iterating over the layers in the model and checking if their name is in the list of layer names
    output_tensors = [model.get_layer(layer).output for layer in layer_names]

    # Create a new model that outputs these tensors given the original model's input
    # This is an 'intermediate' model that allows us to retrieve the outputs of specific layers
    intermediate_model = keras.models.Model(inputs=model.input, outputs=output_tensors)

    # Use this model to predict on the inputs, giving the outputs of the desired layers
    # The output is a list of numpy arrays, one for each layer
    intermediate_outputs = intermediate_model.predict(inputs, verbose=0)

    # Also get the output of the original model
    # This is done by calling predict on the original model
    model_output = model.predict(inputs, verbose=0)

    return model_output, intermediate_outputs


def get_lpfg_inputs(wavelength, lpfg_trans, fbg_pos, optical_source):
    """
    This function prepares the inputs for the LPFG estimation.

    Parameters:
    wavelength (array-like): The wavelength array for simulating the optical setup, given in nm.
    lpfg_trans (array-like): The LPFG transmission transfer function.
    fbg_pos (array-like): The FBG array Bragg wavelengths.
    optical_source (array-like): Optical source used to illuminate the LPFG and FBG

    Returns:
    tuple: The normalized difference between the filtered power and its full spectrum, and the normalized FBG positions.
    """
    # Simulate the FBG interrogator source to get the peaks at FGB position
    source_power = np.interp(fbg_pos, optical_source[:, 0], optical_source[:, 1])
    
    # Get the LPFG transmission power at the FBG positions
    lpfg_filtered = np.interp(fbg_pos, wavelength, lpfg_trans)

    # Get the peaks 
    lpfg_peaks = source_power + lpfg_filtered

    # Calculate the difference between the filtered power and its full spectrum
    input_strength = (lpfg_peaks - source_power)

    # Normalize the difference
    input_strength = input_strength/np.sum(input_strength)

    # To improve robustness to power fluctuation 
    input_strength = input_strength - min(input_strength)
    input_strength = input_strength/np.sum(input_strength)
    
    # Normalize the FBG positions
    fbg_norm = (fbg_pos - 1500)/100

    return input_strength, fbg_pos, fbg_norm


def process_lpfg_batch(wavelength, array_lpfg_trans, fbg_pos, fbg_array, optical_source):
    """
    This function prepares a batch of LPFG tranferfunctions.

    Parameters:
    wavelength (array-like): The wavelength array for simulating the optical setup, given in nm.
    array_lpfg_trans (array-like): The LPFG transmission transfer function array, one per LPFG.
    fbg_pos (array-like): The FBG array Bragg wavelengths.
    fbg_array (array-like): The designed FBG array position. This can be different from fbg_pos since the FBG array could change with temperature and strain.
    optical_source (array-like): Optical source used to illuminate the LPFG and FBG

    Returns:
    tuple: The normalized difference between the filtered power and its full spectrum, and the normalized FBG positions.
    """
    array_input_strength = []
    array_fbgs_distance = []
    for lpfg_trans in array_lpfg_trans:
        
        input_strength, fbgs_distance, fbg_norm = get_lpfg_inputs(wavelength, lpfg_trans, fbg_pos, optical_source)
        array_input_strength.append(input_strength)
        array_fbgs_distance.append(fbgs_distance)
    return np.array(array_input_strength), np.array(array_fbgs_distance)



