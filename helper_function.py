import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
    
# Set a random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv1D, Flatten, Dense, Reshape, Conv1DTranspose
from keras.models import Model
from keras.models import clone_and_build_model
from keras.callbacks import EarlyStopping


def sliding_window(data, window_size, stride):
    """
    Extracts sliding window segments from time-series data for each class label.

    Parameters:
    - data (DataFrame): Input time-series data with columns representing features and a 'Fault_class' column for labels.
    - window_size (int): Size of the sliding window.
    - stride (int): Number of data points to move the sliding window at each step.

    Returns:
    - R (array): Array of sliding window segments.
    - Y (array): Array of corresponding labels for each segment.
    - pseudo_label (array): Array of incidence vector (C) for each segment.
    """
    
    # Initialize variables to store the segments and labels
    
    residuals = []
    labels = []
    C = []
    
    # Iterate over each class label in the data
    for label in data['Fault_class'].unique():
        # Get the data for the current class label
        class_data = data[data['Fault_class'] == label].iloc[30:]
    
        # Iterate over the data
        for i in range(0, len(class_data) - window_size, stride):
            # Get the current segment and label
            
            residual = class_data.iloc[i:i+window_size,1:7]
            
            label = class_data.iloc[i+window_size][-1]

            c = class_data.iloc[i+window_size,7:-1]
        
            # Add the segment and label to the list
            residuals.append(residual)
            labels.append(label)
            C.append(c)
    
    R,Y,pseudo_label= np.array(residuals),  np.array(labels), np.array(C)

    
    # Return the segments and labels
    return R,Y,pseudo_label

def DeepLearningModel(InputFeature, Target, last_layer_activation, loss_fn):
    """
    Create a deep learning model using Convolutional Neural Network (CNN) architecture.

    Parameters:
    - InputFeature (numpy.ndarray): Input features with shape (batch_size, sequence_length, feature_dim).
    - Target (numpy.ndarray): Target values with shape (batch_size, num_classes).
    - last_layer_activation (str): Activation function for the output layer. Common choices: 'sigmoid' for binary classification, 'softmax' for multi-class classification.
    - loss_fn (str): Loss function to optimize during training. Common choices: 'binary_crossentropy' for binary classification, 'categorical_crossentropy' for multi-class classification.

    Returns:
    - nn_model (tensorflow.keras.models.Model): Compiled Keras model for deep learning.

    Example:
    ```python
    # Example usage:
    model = DeepLearningModel(InputFeature, Target, 'sigmoid', 'binary_crossentropy')
    ```
    """
    # Define input layer
    input_layer = Input(shape=(InputFeature.shape[1], InputFeature.shape[2]))

    # Define CNN layers with batch normalization in between
    cnn1 = Conv1D(32, 3, padding='same', activation='relu')(input_layer)
    cnn1 = Conv1D(32, 3, padding='same', activation='relu')(cnn1)
    cnn1 = Flatten()(cnn1)

    # Define hidden layer with batch normalization
    hidden_layer = Dense(units=64, activation='relu')(cnn1)

    # Define output layer with specified activation function
    outputs = Dense(Target.shape[1], activation=last_layer_activation)(hidden_layer)

    # Define the model
    nn_model = Model(inputs=input_layer, outputs=outputs)

    # Compile the model with specified loss function and Adam optimizer
    nn_model.compile(loss=loss_fn, optimizer='adam', metrics=['binary_accuracy'])

    return nn_model

def Resample(X_sc, Y, Z, num_samples):
    """
    Resample the input data to balance the class distribution.

    Parameters:
    - X_sc (numpy.ndarray): Input data array. If 2D, it represents features; if 3D, it represents features over time steps.
    - Y (numpy.ndarray): Array of labels corresponding to each data point in X_sc.
    - Z (numpy.ndarray): Additional information array with the same number of features as X_sc.
    - num_samples (int): Number of samples to be resampled for each unique class label.

    Returns:
    - tuple: A tuple containing resampled data arrays (x_sample, y_sample, z_sample).
      - x_sample (numpy.ndarray): Resampled input data array.
      - y_sample (numpy.ndarray): Resampled array of labels.
      - z_sample (numpy.ndarray): Resampled additional information array.

    Notes:
    - The function uses sklearn.utils.resample to resample data for each unique class label.
    - Assumes Z has the same number of features as X_sc.

    Example:
    >>> X_resampled, Y_resampled, Z_resampled = resample(X_train, Y_train, Z_train, num_samples=1000)
    """
    # Check if the input data is 2D or 3D
    if len(X_sc.shape) == 3:
        num_features = X_sc.shape[1]
        num_time_steps = X_sc.shape[2]
        x_sample = np.zeros((num_samples * np.unique(Y).size, num_features, num_time_steps))
    else:
        num_features = X_sc.shape[1]
        x_sample = np.zeros((num_samples * np.unique(Y).size, num_features))
    y_sample = np.zeros(num_samples * np.unique(Y).size)
    z_sample = np.zeros((num_samples * np.unique(Y).size, Z.shape[1]))  # Assuming Z has the same number of features as X

    for i, label in enumerate(np.unique(Y)):
        class_indices = np.where(Y == label)[0]
        sampled_indices = resample(class_indices, n_samples=num_samples, replace=False, random_state=0)
        x_sample[i*num_samples:(i+1)*num_samples] = X_sc[sampled_indices]
        y_sample[i*num_samples:(i+1)*num_samples] = Y[sampled_indices]
        z_sample[i*num_samples:(i+1)*num_samples] = Z[sampled_indices]

    return x_sample, y_sample, z_sample


def FineTunedModel(nn_model,x_train,y_train,X_sc,Y_ohe):
    """
    Fine-tunes a neural network model by adding and training additional layers on top of
    the intermediate layer of the provided base model.

    Parameters:
    - nn_model (tf.keras.Model): The base neural network model to be fine-tuned.
    - x_train (numpy.ndarray): The input training data.
    - y_train (numpy.ndarray): The target training data.
    - X_sc (numpy.ndarray): The input validation data for model evaluation.
    - Y_ohe (numpy.ndarray): The target validation data for model evaluation, one-hot encoded.

    Returns:
    - tf.keras.Model: The fine-tuned model.

    The function performs the following steps:
    1. Creates a clone of the base model and sets its weights.
    2. Constructs an intermediate model by removing the last two layers from the clone.
    3. Freezes the layers of the intermediate model.
    4. Adds new layers to the intermediate model for fine-tuning.
    5. Compiles the model with categorical cross-entropy loss and the Adam optimizer.
    6. Trains the model on the training data with freezed layers for 500 epochs.
    7. Unfreezes the layers for further training.
    8. Creates a custom Adam optimizer with a small learning rate.
    9. Recompiles the model with the custom optimizer.
    10. Retrains the model on the training data with unfreezed layers for 50 epochs.
    11. Returns the fine-tuned model.

    Note:
    The function uses early stopping with patience set to 5 to monitor validation loss
    and restore the best weights during training.
    """

    from keras.layers import BatchNormalization
    from keras.models import clone_and_build_model

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model_copy= clone_and_build_model(nn_model)
    model_copy.set_weights(weights=nn_model.get_weights())

    intermediate_model = Model(inputs=model_copy.input,outputs=model_copy.layers[-2].output)

    for l in intermediate_model.layers:
        l.trainable=False

    fine_tuned_layers = Dense(units=64,activation='relu')(intermediate_model.output)
    output_layer = Dense(units=Y_ohe.shape[1],activation='softmax')(fine_tuned_layers)

    # Define the model
    fine_tuned_model = Model(inputs=intermediate_model.input, outputs=output_layer)

    # Compile the model with binary cross-entropy loss function and Adam optimizer
    fine_tuned_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #categorical_hinge

    #train the model with freezed layer
    history1=fine_tuned_model.fit(x_train, y_train, epochs=500, batch_size=int(len(x_train) * 0.2), validation_data=(X_sc[::50], Y_ohe[::50]), callbacks=[early_stop], verbose=0)

    for l in fine_tuned_model.layers:
        l.trainable=True

    # Create a custom Adam optimizer with a small learning rate
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    # Compiling again is necessary to update the trainable parameter before training
    fine_tuned_model.compile(loss='categorical_crossentropy', optimizer=custom_optimizer, metrics=['accuracy']) #categorical_hinge

    #train the model with unfreezed layer
    history2=fine_tuned_model.fit(x_train, y_train, epochs=50, batch_size=int(len(x_train) * 0.2), validation_data=(X_sc[::50], Y_ohe[::50]), verbose=0)

    return fine_tuned_model

def preprocess_inferance(data,scaler,window_size, stride):
    """
    Preprocesses input data for inference.

    Args:
    - data (DataFrame): Input DataFrame containing time-series data.
    - scaler: A scaler object (e.g., from scikit-learn) for normalizing the input data.
    - window_size (int): Size of the sliding window used to extract segments from the time-series data.
    - stride (int): The step size for moving the sliding window.

    Returns:
    - R (numpy.ndarray): Processed residual segments.
    - Y (numpy.ndarray): Corresponding labels for each segment.
    - C (numpy.ndarray): Additional features (context) for each segment.
    - Time (numpy.ndarray): Timestamps corresponding to each segment.

    This function iterates over the input data and extracts segments of the specified window size
    with a specified stride. It normalizes the segments using the provided scaler and returns the
    processed residuals (R), labels (Y), context features (C), and timestamps (Time).
    """
    residuals = []
    labels = []
    Time = []
    C = []
    # Iterate over the data
    for i in range(0, len(data) - window_size, stride):
        # Get the current segment and label
        
        residual = data.iloc[i:i+window_size,1:7]
        
        label = data.iloc[i+window_size][-1]

        c = data.iloc[i+window_size,7:-1]

        time = data.iloc[i+window_size,0]

        # Add the segment and label to the list
        residuals.append(residual)
        labels.append(label)
        C.append(c)
        Time.append(time)

    R,Y,C,Time= np.array(residuals),  np.array(labels), np.array(C), np.array(Time)

    R=scaler.transform(R.reshape(-1,R.shape[2])).reshape(R.shape)

    return R,Y,C,Time



def Cascade_Hybrid_FDI(Z,y_pred):
    """
    Perform Cascade Hybrid Fault Detection and Isolation (FDI).

    This function combines the predictions from a hybrid FDI system based on a set of fault signals Z
    and the predicted output values y_pred.

    Parameters:
    - Z (array-like): A 2D array representing the fault signals. Each row corresponds to a sample,
                     and each column represents a different fault signal. Non-zero values indicate the presence
                     of a fault.
    - y_pred (array-like): A 2D array representing the predicted output values. Each row corresponds to a sample,
                          and each column represents a different output dimension.

    Returns:
    - Cascade_hybrid_prediction (array-like): An array containing the final predictions after Cascade Hybrid FDI.
                                              If a fault is detected in a sample (sum of the corresponding row in Z is > 0),
                                              the prediction is taken from the first column of y_pred (index 0).
                                              Otherwise, the prediction is set to 0.
    """
    Cascade_hybrid_prediction=np.zeros(len(y_pred))

    for i,fs in enumerate(Z):
        if sum(fs)>0:
            Cascade_hybrid_prediction[i]=y_pred[i]

    return Cascade_hybrid_prediction



def Visualize_last_layer(model,X_sc,Y,Z,last_layer=-1):
    X_sc,Y,_ = Resample(X_sc,Y,Z,50)
  
    intermediate_model = Model(inputs=model.input,outputs=model.layers[last_layer].output)

    from sklearn.manifold import TSNE
    t_sne = TSNE(n_components=2,perplexity=20)

    x_inter = intermediate_model.predict(X_sc)
    x_embedded = t_sne.fit_transform(x_inter)
    y_label = Y

    # Create a scatter plot of the embedded data, colored by the true labels
    f, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(x=x_embedded[:,0],y=x_embedded[:,1],hue=y_label,style=y_label,palette="bright",edgecolor='black',alpha=0.5)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title('After training')
    plt.show()
   