from typing import Optional, Iterable, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import date
import random

from time_series_anomaly_detection.abstractions import (
    TimeSeriesAnomalyDetector
)

class GAN_AD(TimeSeriesAnomalyDetector):
    """
    Time series GAN anomaly detector.
    Parameters
    ----------
    id_columns: Iterable[str], optional
        ID columns used to identify individual time series.
        Should be specified in case the detector is provided with
        time series during training or inference with ID columns
        included. Using these columns the detector can separate individual
        time series and not use ID columns as feature columns.
        In case they are not specified, all columns are regarded as feature
        columns and the provided data is regarded as a single time series.
    """
    def __init__(
            self,
            window_size: int = 8,
            shift: int = 3,
            batch_size: int = 8,
            latent_dim: int = 1,
            id_columns: Optional[Iterable[str]] = None
    ):
        """
        Inits GAN_AD class with parameters which are related to
        the training process and data supplied later.
        Parameters
        ----------
        window_size : int, default 8
            Length of the window taken across time series sequences.
            The window is applied to both train and test data in order
            to subdivide long sequences into smaller time series.
        shift : int, default 3
            Length of window shift applied to both train in order
            to subdivide long sequences into smaller time series.
        batch_size : int, default 8
            The number of time series samples in training dataset batch.
        latent_dim : int, default 1
            Latent space dimension used during GAN-AD model training.
        id_columns : Iterable[str], default None
            Names of ID columns used to separate individual time series.
        """
        super().__init__()
        
        self._window_size = window_size
        self._shift = shift
        self._batch_size = batch_size
        self._latent_dim = latent_dim
        self._id_columns = id_columns
        
        self._scaler = StandardScaler()
        self._bce = BinaryCrossentropy()
        
        
    def _build_discriminator(self) -> keras.Model:
        """
        Method used to construct the discriminator network.
        
        Returns
        -------
        keras.Model
            Discriminator of GAN-AD model.
        """
        data = Input(shape=(self._window_size, self._n_features))
        
        x = LSTM(150)(data)
        x = Dense(50)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        decision = Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=[data], outputs=[decision], name='discriminator')
        return model
    
    def _build_generator(self) -> keras.Model:
        """
        Method used to construct the generator network.
        
        Returns
        -------
        keras.Model
            Generator of GAN-AD model.
        """
        z = Input(shape=(self._window_size, self._latent_dim))
        
        x = Conv1D(30, padding='same', kernel_size=3)(z)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = LSTM(200)(z)
        x = Dense(self._window_size * self._n_features, activation='linear')(x)
        
        series = Reshape((self._window_size, self._n_features))(x)
        model = keras.Model(inputs=[z], outputs=[series], name='generator')
        return model
    
    def _build_gan_ad(self) -> keras.Model:
        """
        Merges both components of GAN-AD together into one model.
        
        Returns
        -------
        keras.Model
            GAN-AD model.
        
        """
        generator = self._build_generator()
        discriminator = self._build_discriminator()
        
        z = Input(shape=(self._window_size, self._latent_dim))
        generated_series = generator([z])
        generated_series_eval = discriminator([generated_series])
        
        model = keras.Model(inputs=[z], outputs=[generated_series, generated_series_eval], name='gan')
        return model
    
    def _get_discriminator_loss(self, real: tf.Tensor, generated: tf.Tensor) -> tf.Tensor:
        """
        Computes loss of the discriminator using its output opinion for 
        real time series samples and samples obtained from the generator.
        
        Parameters
        ----------
        real : tf.Tensor
            Discriminator's probability opinion of real time series data.
        generated : tf.Tensor
            Discriminator's probability opinion of fake time series data
            obtained from the generator.
        Returns
        -------
        tf.Tensor
            Discriminator's loss.
        """
        # loss on real time series data
        real_loss = self._bce(tf.ones_like(real), real)
        # loss on fake time series data
        generated_loss = self._bce(tf.zeros_like(generated), generated)
        
        d_loss = real_loss + generated_loss
        return d_loss
    
    def _get_generator_loss(self, generated: tf.Tensor) -> tf.Tensor:
        """
        Computes loss of the generator using generated time series samples
        evaluated by the discriminator.
        
        Parameters
        ----------
        generated : tf.Tensor
            Discriminator's probability opinion of fake time series data
            obtained from the generator.
            
        Returns
        -------
        tf.Tensor
            Generator's loss.
        """
        return self._bce(tf.ones_like(generated), generated)
    
    @tf.function
    def _fit_batch(self, batch: tf.Tensor, z: tf.Tensor) -> Tuple:
        """
        Trains model with a single batch of real time series data
        and a single batch of generated latent vectors.        
        This function performs a single weight update for both of 
        the networks.
        
        Parameters
        ----------
        batch : tf.Tensor
            A single batch of real time series data.
        z : tf.Tensor
            A single batch of generated latent vectors.
            
        Returns
        -------
        Tuple
            Discriminator's and generator's loss.
        """
        with tf.GradientTape(persistent=True) as gt:
            generated_series = self._generator(z, training=True)
            generated_series_eval = self._discriminator(generated_series, training=True)
            real_series_eval = self._discriminator(batch, training=True)
            g_loss = self._get_generator_loss(generated_series_eval)
            d_loss = self._get_discriminator_loss(real_series_eval, generated_series_eval)
            
        # compute the gradients
        discriminator_gradients = gt.gradient(d_loss, self._discriminator.trainable_variables)
        generator_gradients = gt.gradient(g_loss, self._generator.trainable_variables)
        
        # apply the gradients
        self._discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self._discriminator.trainable_variables))
        self._generator_optimizer.apply_gradients(zip(generator_gradients, self._generator.trainable_variables))
        return (d_loss, g_loss)
    
    def save_weights(self, file_name: str) -> None:
        """
        Saves weights of the GAN-AD model.
        
        Parameters
        ----------
        file_name : str
            Name of the file to save the model's weights to.
        """
        self._gan_ad.save_weights(file_name)
        
    def load_weights(self, file_name: str) -> None:
        """
        Loads all layer weights to the GAN-AD model.
        
        Parameters
        ----------
        file_name : str
            Name of the file to load the weights from.
        """
        self._gan_ad.load_weights(file_name)
        
    def _train(self, dataset: tf.data.Dataset, epochs: int, save_checkpoints: Optional[bool] = False,
               enable_prints: Optional[bool] = False) -> None:
        """
        Trains the GAN-AD model on a dataset for a specific number of epochs,
        both supplied as input parameters.
        
        Parameters
        ----------
        dataset : tf.data.Dataset
            Time series training data.
        epochs : int
            Number of iterations over the dataset to train the GAN-AD model.
        save_checkpoints : Optional[bool], default False
            Value determinating whether to periodically save the model
            during the training or not.
        enable_prints : Optional[bool], default False
            Value determinating whether to print the training progress or not.
        """
        loss_history = []
        for epoch in range(epochs):
            batch_loss_history = []
            
            for batch in dataset:
                z = tf.random.normal([self._batch_size, self._window_size, self._latent_dim])
                d_loss, g_loss = self._fit_batch(batch, z)
                batch_loss_history.append([d_loss.numpy(), g_loss.numpy()])
            loss_history.append(np.mean(batch_loss_history, axis=0))
            
            # if enable_prints is True, print the progress of calculated losses, 
            # real vs generated time series window and grid of nine generated
            # time series windows
            if ((epoch != 0) and (epoch % 500 == 0) and enable_prints):
                print(epoch, ". epoch")
                print("Batch loss [disc_loss, gen_loss] :", np.mean(batch_loss_history, axis=0))
                plt.plot(np.array([loss[0] for loss in loss_history]), color='green')
                plt.plot(np.array([loss[1] for loss in loss_history]), color='blue')
                plt.show()
                
                print('Real window')
                window = pd.DataFrame(dataset.__iter__().__next__()[0].numpy())
                window.plot(figsize=(10, 5))
                plt.show()
                
                z = tf.random.normal([self._batch_size, self._window_size, self._latent_dim])
                generated_series = self._generator(z, training=True)
                print('Generated window')
                window = pd.DataFrame(generated_series[0].numpy())
                window.plot(figsize=(10, 5))
                plt.show()
                
                print('Generated windows - possible mode collapse visualisation')
                fig, axs = plt.subplots(3, 3, figsize=(15, 15))
                for i in range(9):
                    z_init = tf.random.normal([self._window_size, self._latent_dim])
                    generated_sample = self._generator(tf.reshape(z_init, [1, self._window_size, self._latent_dim]), training=False)[0]
                    axs[int(i % 3), int(i / 3)].plot(generated_sample.numpy())
                plt.show()
                print()
                
            # if save_checkpoints is True, continuously save the model's weights
            if (save_checkpoints and (epoch != 0) and (epoch % 1000 == 0)):
                self.save_weights("gan_ad_" + str(date.today()) + "_epoch_" + str(epoch) + ".h5")
                
    def fit_scaler(self, data: pd.DataFrame) -> None:
        """
        Computes the mean and standard deviation from the supplied data
        to be used later for scaling.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data.
        """
        self._scaler.fit(data)
        
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes the data by centering and scaling using
        mean and standard deviation stored in a previously fitted scaler.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to standardize.
        
        Returns
        -------
        pd.DataFrame
            Standardized data.
        """
        data = self._scaler.transform(data)
        return pd.DataFrame(data)
    
    def _get_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Standardizes the data and creates a TensorFlow dataset of sliding windows
        over an individual time series provided as the input parameter.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data to be preprocessed into a TensorFlow dataset.
            
        Returns
        -------
        tf.data.Dataset
            Time series TensorFlow dataset.
        """
        data_scaled = pd.DataFrame(self._scaler.fit_transform(data))
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_scaled.astype(np.float),
                                                                       targets=None, sequence_length=self._window_size,
                                                                       sequence_stride=self._shift, sampling_rate=1,
                                                                       batch_size=self._batch_size, shuffle=True,
                                                                       seed=6)
        return dataset
    
    def _preprocess_data(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Preprocesses time series identified by the id_columns.
        In case of multiple time series, the function separates the input 
        data into several individual time series.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data to be preprocessed into TensorFlow dataset.
            Must contain all of the columns specified in id_columns.
            
        Returns
        -------
        tf.data.Dataset
            Time series TensorFlow dataset.
        """
        if (self._id_columns == None):
            dataset = self._get_dataset(data)
        else:
            dfs = [x.reset_index(drop=True) for _, x in data.groupby(self._id_columns)]
            [df.drop(columns=self._id_columns, inplace=True) for df in dfs]
            datasets = [self._get_dataset(df) for df in dfs]
            dataset = datasets.pop(0)
            for single_dataset in datasets:
                dataset = dataset.concatenate(single_dataset)
            dataset = dataset.shuffle(buffer_size=len(data), seed=6)
        return dataset
    
    def fit(self, X: pd.DataFrame, n_epochs: int = 10, d_learning_rate: float = 0.0002,
            g_learning_rate: float = 0.00002, save_checkpoints: Optional[bool] = False,
            enable_prints: Optional[bool] = False, *args, **kwargs) -> None:
        """
        Fits the GAN-AD model according to the given training data and hyperparameter
        setting. Function also allows to enable prints and saving checkpoints
        during the model's training.
        
        Parameters
        ----------
        X : pd.DataFrame
            The raw training data. The columns contain features and
            possibly also identifiers of individual time series (given in id_columns).
        n_epochs : int, default 10
            Number of iterations over dataset to train the GAN-AD model.
        d_learning_rate : float, default 0.0002
            Learning rate of optimizer used for the discriminator.
        g_learning_rate : float, default 0.00002
            Learning rate of optimizer used for the generator.
        save_checkpoints : Optional[bool], default False
            Enables or forbids saving checkpoints during the training.
        enable_prints : Optional[bool], default False
            Enables or forbids printing the training progress.
        """
        id_cols = len(self._id_columns) if self._id_columns is not None else 0
        self._n_features = X.shape[1] - id_cols
        
        # build GAN-AD model architecture
        self._gan_ad = self._build_gan_ad()
        self._discriminator = self._gan_ad.get_layer('discriminator')
        self._generator = self._gan_ad.get_layer('generator')
        
        dataset = self._preprocess_data(X)
        
        # initialize optimizers for discriminator and generator network
        self._discriminator_optimizer = RMSprop(learning_rate=d_learning_rate)
        self._generator_optimizer = Adam(learning_rate=g_learning_rate, beta_1=0.5)
        
        # train GAN-AD model
        self._train(dataset, n_epochs, save_checkpoints, enable_prints)
        pass
    
    def save_model(self, file_name: str) -> None:
        """
        Saves the GAN-AD model.
        
        Parameters
        ----------
        file_name : str
            Name of the file to save the model to.
        """
        self._gan_ad.save(file_name)
        
    # Implementation from https://github.com/ananda1996ai/MMD-AutoEncoding-GAN-MAEGAN-/blob/master/maegan_final.py
    # Maximum Mean Discrepancy Auto-Encoding Generative Adversarial Networks (MAEGAN)
    def _rbf_kernel(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Calculates the Radial Basis Function (RBF) kernel.
        
        Parameters
        ----------
        x : tf.Tensor
            Vector to be compared (represents a single time series window).
        y : tf.Tensor
            Vector to be compared (represents a single time series window).
        
        Returns
        -------
        tf.Tensor
            Similarity of vectors computed using RBF kernel function.
            
        """
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        
        dim = tf.shape(x)[1]
        
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float64))
    
    def _get_mmd_similarity(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Computes the similarity of two input vectors.
        Currently it is only possible to use the RBF kernel. 
        
        Parameters
        ----------
        x : tf.Tensor
            Vector to be compared (represents a single time series window).
        y : tf.Tensor
            Vector to be compared (represents a single time series window).
            
        Returns
        -------
        tf.Tensor
            Similarity of the input vectors.
        """
        xy_kernel = self._rbf_kernel(x, y)
        return xy_kernel
    
    def _find_mapping(self, sample: tf.Tensor) -> tf.Tensor:
        """
        Finds the (sub)optimal latent space representation of the time
        series window sample on the input. Algorithm iteratively improves 
        the latent representation by maximizing the similarity between 
        the original time series window and the window obtained from 
        the generator fed with the best yet found latent representation.
        
        Parameters
        ----------
        sample : tf.Tensor
            Time series window from the dataset whose latent space 
            representation is being searched for.
            
        Returns
        -------
        tf.Tensor
            Latent space representation corresponding to the input.
        """
        # error is considered small enough if its smaller than tolerance
        tolerance = 0.05
        
        # sample random latent representation
        z_init = tf.random.normal([self._window_size, self._latent_dim])
        
        z_opt = tf.Variable(z_init, trainable=True, dtype=tf.float32)
        mse = tf.keras.losses.MeanSquaredError()
        
        def get_loss():
            generated_sample = self._generator(tf.reshape(z_opt, [1, self._window_size, self._latent_dim]), training=False)[0]
            similarity_per_sample = self._get_mmd_similarity(tf.cast(sample, tf.float64),
                                                             tf.cast(generated_sample, tf.float64))
            reconstruction_loss_per_sample = 1 - similarity_per_sample
            return reconstruction_loss_per_sample
        
        # iteratively improve the latent representation of the input sample
        for i in range(1000):
            RMSprop(learning_rate=0.01).minimize(get_loss, var_list=[z_opt])
            if (tf.reduce_mean(get_loss()) < tolerance):
                break
        
        return z_opt
    
    def _replace_NaN_values(self, X: pd.DataFrame, rows_with_NaN: pd.Index) -> pd.DataFrame:
        """
        Replaces NaN values in the inputed DataFrame by the mean values
        computed featurewise from the non-NaN values.
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series input data with NaNs to be replaced.
        rows_with_NaN : pd.Index
            Indexes of time series records containing at least one NaN value.
            
        Returns
        -------
        pd.DataFrame
            Time series DataFrame with NaNs replaced with mean values.
        """
        mean_vals = pd.DataFrame(X.mean())
        for row in rows_with_NaN:
            X.iloc[[row]] = np.transpose(mean_vals).iloc[[0], :].values
        return X
        
    def predict_window_anomaly_scores(self, X: pd.DataFrame) -> tf.Tensor:
        """
        Predicts an anomaly score of the time series window for each timestep.
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series window for which the anomaly scores are to be predicted.
            
        Returns
        -------
        tf.Tensor
            Predicted anomaly scores.
        """
        z_mapping = self._find_mapping(X)
        lamda = 0.5
        
        # calculate the residuals
        generated_sample = self._generator(tf.reshape(z_mapping, [1, self._window_size, self._latent_dim]), training=False)[0]
        residuals = tf.math.reduce_sum(abs(tf.cast(X, tf.float64) - tf.cast(generated_sample, tf.float64)), axis=1)

        # calculate the discrimination results
        real_prob = self._discriminator(tf.reshape(generated_sample, [1, generated_sample.shape[0], generated_sample.shape[1]]),
                                        training=False)[0]
        
        
        # the equation for the anomaly score is slightly modified and inspired by the following paper
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8942842&tag=1
        # the main difference from our paper is that the second part of the equation
        # is not weighted output from the discriminator, but weighted (1 - output from the discriminator) instead 
        
        # obtain anomaly score
        anomaly_score = tf.add((1 - lamda) * residuals, tf.cast(lamda * (1 - real_prob), tf.float64))
        return anomaly_score
    
    def predict_series_anomaly_scores(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts an anomaly score of the time series on the input for each timestep.
        Individual time series are being processed by parts. The input data
        are divided into smaller time series with the length of a window size.
        Incomplete windows are filled with NaN values. For the input data
        of insufficient size (less than one window) we return a Series of NaN values.
        
        Parameters
        ----------
        X : pd.DataFrame
            Individual time series for which the anomaly scores are predicted.
            
        Returns
        -------
        pd.Series
            Predicted anomaly scores.
        """
        
        # not enough data to predict anomaly scores
        if (X.shape[0] < self._window_size):
            nan_array = np.empty(X.shape[0])
            nan_array[:] = np.nan
            return pd.Series(nan_array)
        
        resid = X.shape[0] % self._window_size
        
        # check if there are any NaN values
        is_NaN = X.isnull()
        row_has_NaN = is_NaN.any(axis=1)
        rows_with_NaN = X[row_has_NaN]
        
        # fill the NaN values, so the generator could work properly
        if (not rows_with_NaN.empty):
            X = self._replace_NaN_values(X, rows_with_NaN.index)
            
        X = self._scaler.transform(X)
        
        windows = [pd.DataFrame(X[i:i + self._window_size]) for i in range(0, X.shape[0] - resid, self._window_size)]
        anomaly_score = tf.concat([self.predict_window_anomaly_scores(X_window) for X_window in windows], -1)
        anomaly_score = pd.Series(anomaly_score).copy()
        
        # reinsert NaN values
        if (not rows_with_NaN.empty):
            anomaly_score.at[rows_with_NaN.index] = np.nan
        
        # fill the last incomplete window with NaN values
        if (resid != 0):
            nan_array = np.empty(resid)
            nan_array[:] = np.nan
            nan_series = pd.Series(nan_array)
            anomaly_score = pd.concat([anomaly_score, nan_series], axis=0).reset_index(drop=True)
            
        return anomaly_score
    
    def predict_anomaly_scores(self, X: pd.DataFrame, *args, **kwargs) -> pd.Series:
        """
        Predicts an anomaly score of the time series on the input for each timestep. 
        The higher the score, the more abnormal the measured timestep sample is.
        Time series is being processed by window-sized parts. In the case of multiple 
        time series provided on the input, the time series is split to several individual
        time series using a id_columns.
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series for which the anomaly scores are to be predicted.
        
        Returns
        -------
        pd.Series
            Predicted anomaly scores.
        """
        if (self._id_columns == None):
            anomaly_score = self.predict_series_anomaly_scores(X)
        else:
            dfs = [x.reset_index(drop=True) for _, x in X.groupby(self._id_columns)]
            [df.drop(columns=self._id_columns, inplace=True) for df in dfs]
            anomaly_score = pd.concat([self.predict_series_anomaly_scores(df) for df in dfs], axis=0).reset_index(drop=True)
            
        return anomaly_score
    
    def identify_anomaly(self, X: pd.DataFrame, threshold: Optional[int] = 1) -> np.array:
        """
        Identifies anomalies in the input time series. An anomaly is detected
        if the cross entropy of the anomaly score exceeds the threshold.
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series for which the anomalies are to be predicted.
        threshold : Optional[int], default 1
            Value which determinates if the timestamp is going to be 
            flaged as an anomaly.
            
        Returns
        -------
        np.array
            Identified timesteps where the anomalies appear.
        """
        anomaly_score = self.predict_anomaly_scores(X)
        cross_entropy = np.log(anomaly_score)
        identified = (cross_entropy > threshold).astype(int)
        
        return identified