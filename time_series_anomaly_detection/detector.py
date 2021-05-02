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
        window_size: int,
        shift: int,
        batch_size: int,
        latent_dim: int,
        n_features: int,
        id_columns: Optional[Iterable[str]] = None
    ):
        super().__init__()
        self._window_size = window_size
        self._shift = shift
        self._batch_size = batch_size
        self._latent_dim = latent_dim
        self._n_features = n_features
        self._id_columns = id_columns
        
        self._scaler = StandardScaler()
        self._bce = BinaryCrossentropy()
        
        self._gan_ad = self._build_gan_ad()
        self._discriminator = self._gan_ad.get_layer('discriminator')
        self._generator = self._gan_ad.get_layer('generator')
    
    def _build_discriminator(self) -> keras.Model:
        data = Input(shape=(self._window_size, self._n_features))

        x = LSTM(150)(data)
        x = Dense(50)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        decision = Dense(1, activation = 'sigmoid')(x)

        model = keras.Model(inputs=[data], outputs=[decision], name='discriminator')
        return model
    
    def _build_generator(self) -> keras.Model:
        z = Input(shape=(self._window_size, self._latent_dim))

        x = Conv1D(30, padding='same', kernel_size=3) (z)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = LSTM(200)(z)
        x = Dense(self._window_size*self._n_features, activation = 'linear')(x)

        series = Reshape((self._window_size, self._n_features))(x)

        model = keras.Model(inputs=[z], outputs=[series], name='generator')
        return model
    
    def _build_gan_ad(self) -> keras.Model:
        generator = self._build_generator()
        discriminator = self._build_discriminator()

        z = Input(shape=(self._window_size, self._latent_dim))

        generated_series = generator([z])
        generated_series_eval = discriminator([generated_series])

        model = keras.Model(inputs=[z], outputs=[generated_series, generated_series_eval], name='gan')
        return model
    
    def _get_discriminator_loss(self, real: tf.Tensor, generated: tf.Tensor) -> tf.Tensor:
        real_loss = self._bce(tf.ones_like(real), real)
        generated_loss = self._bce(tf.zeros_like(generated), generated)
        d_loss = real_loss + generated_loss
        return d_loss
    
    def _get_generator_loss(self, generated: tf.Tensor) -> tf.Tensor:
        return self._bce(tf.ones_like(generated), generated)
    
    @tf.function
    def _fit_batch(self, batch: tf.Tensor, z: tf.Tensor) -> Tuple:
        with tf.GradientTape(persistent=True) as gt:

            generated_series = self._generator(z, training = True)

            generated_series_eval = self._discriminator(generated_series, training = True)
            real_series_eval = self._discriminator(batch, training = True)

            g_loss = self._get_generator_loss(generated_series_eval)
            d_loss = self._get_discriminator_loss(real_series_eval, generated_series_eval)

        discriminator_gradients = gt.gradient(d_loss, self._discriminator.trainable_variables)
        generator_gradients = gt.gradient(g_loss, self._generator.trainable_variables)

        self._discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self._discriminator.trainable_variables))
        self._generator_optimizer.apply_gradients(zip(generator_gradients, self._generator.trainable_variables))

        return (d_loss, g_loss)
    
    def save_weights(self, file_name:str) -> None:
        self._gan_ad.save_weights(file_name)
       
    def load_weights(self, file_name:str) -> None:
        self._gan_ad.load_weights(file_name)
    
    def _train(self, dataset, epochs: int, save_checkpoints: Optional[bool] = False,
               enable_prints: Optional[bool] = False) -> None:
        loss_history = []
        for epoch in range(epochs):
            batch_loss_history = []
            for batch in dataset:
                z = tf.random.normal([self._batch_size, self._window_size, self._latent_dim])
                d_loss, g_loss = self._fit_batch(batch, z)
                batch_loss_history.append([d_loss.numpy(), g_loss.numpy()])
            loss_history.append(np.mean(batch_loss_history, axis = 0))
            if ((epoch != 0) and (epoch%500 == 0) and enable_prints):
                print(epoch, ". epoch")
                print("Batch loss [disc_loss, gen_loss] :", np.mean(batch_loss_history, axis=0))
                plt.plot(np.array([loss[0] for loss in loss_history]), color='green')
                plt.plot(np.array([loss[1] for loss in loss_history]), color='blue')
                plt.show()

                print('Real window')
                window = pd.DataFrame(dataset.__iter__().__next__()[0].numpy())
                window.plot(figsize=(10,5))
                plt.show()

                z = tf.random.normal([self._batch_size, self._window_size, self._latent_dim])
                generated_series = self._generator(z, training = True)
                print('Generated window')
                window = pd.DataFrame(generated_series[0].numpy())
                window.plot(figsize=(10,5))
                plt.show()

                print('Generated windows - possible mode collapse visualisation')
                fig, axs = plt.subplots(3, 3,figsize=(15,15))
                for i in range(9):
                    z_init = tf.random.normal([self._window_size, self._latent_dim])
                    generated_sample = self._generator(tf.reshape(z_init, [1, self._window_size, self._latent_dim]), training = False)[0]
                    axs[int(i%3), int(i/3)].plot(generated_sample.numpy())
                plt.show()
                print()
            if (save_checkpoints and (epoch != 0) and (epoch%1000 == 0)):
                self.save_weights("gan_ad_" + str(date.today()) + "_epoch_" + str(epoch) + ".h5")
                
    def fit_scaler(self, data: pd.DataFrame) -> None:
        self._scaler.fit(data)
        
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._scaler.transform(data) 
        return pd.DataFrame(data)
    
    def _get_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        
        data_scaled = pd.DataFrame(self._scaler.fit_transform(data))
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data_scaled.astype(np.float),
                                                               targets=None, sequence_length=self._window_size,
                                                               sequence_stride=self._shift, sampling_rate=1,
                                                               batch_size=self._batch_size, shuffle=True, seed=6)
        return dataset
                
    def _preprocess_data(self, data: pd.DataFrame) -> tf.data.Dataset:
        
        if(self._id_columns == None):
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
    
    def fit(self, X: pd.DataFrame, n_epochs: int, d_learning_rate: float = 0.0002,
            g_learning_rate: float = 0.00002, save_checkpoints: Optional[bool] = False,
            enable_prints: Optional[bool] = False, *args, **kwargs) -> None:
        
        dataset = self._preprocess_data(X)
        
        self._discriminator_optimizer = RMSprop(learning_rate=d_learning_rate)
        self._generator_optimizer = Adam(learning_rate=g_learning_rate, beta_1=0.5) 

        self._train(dataset, n_epochs, save_checkpoints, enable_prints)
        pass
    
    def save_model(self, file_name: str) -> None:
        self._gan_ad.save(file_name)
    
    # implementation from https://github.com/ananda1996ai/MMD-AutoEncoding-GAN-MAEGAN-/blob/master/maegan_final.py
    # Maximum Mean Discrepancy Auto-Encoding Generative Adversarial Networks (MAEGAN)
    def _rbf_kernel(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]

        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))

        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float64))

    def _get_mmd_similarity(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        xy_kernel = self._rbf_kernel(x, y)
        return xy_kernel
    
    def _find_mapping(self, sample: tf.Tensor) -> tf.Tensor:
        tolerance = 0.05

        z_init = tf.random.normal([self._window_size, self._latent_dim])
        z_opt = tf.Variable(z_init, trainable=True, dtype=tf.float32)

        mse = tf.keras.losses.MeanSquaredError()

        def get_loss():
            generated_sample = self._generator(tf.reshape(z_opt, [1, self._window_size, self._latent_dim]), training = False)[0]
            similarity_per_sample = self._get_mmd_similarity(tf.cast(sample, tf.float64), tf.cast(generated_sample, tf.float64))
            reconstruction_loss_per_sample = 1 - similarity_per_sample
            return reconstruction_loss_per_sample

        for i in range(1000):
            RMSprop(learning_rate=0.01).minimize(get_loss, var_list=[z_opt])
            if(i%200 == 0):
                tf.print('loss celkovo: ', tf.reduce_mean(get_loss()))
            if(tf.reduce_mean(get_loss()) < tolerance):
                break

        tf.print('final loss: ', tf.reduce_mean(get_loss()))
        tf.print(z_opt)
        return z_opt
    
    def predict_window_anomaly_scores(
            self, X: pd.DataFrame, *args, **kwargs
        ) -> tf.Tensor:
        
        z_mapping = self._find_mapping(X)
        lamda = 0.5

        # calculate the residuals
        generated_sample = self._generator(tf.reshape(z_mapping, [1, self._window_size, self._latent_dim]), training = False)[0]
        
        print('sample')
        window = pd.DataFrame(X)
        window.plot(figsize=(10,5))
        plt.show()
        
        print('generated_sample')
        window = pd.DataFrame(generated_sample.numpy())
        window.plot(figsize=(10,5))
        plt.show()

        residuals = tf.math.reduce_sum(abs(tf.cast(X, tf.float64) - tf.cast(generated_sample, tf.float64)), axis=1)
        print('residuals: ')
        print(residuals)

        # calculate the discrimination results
        real_prob = self._discriminator(tf.reshape(generated_sample, [1, generated_sample.shape[0], generated_sample.shape[1]]),
                                        training = False)[0]
        print('real_prob')
        print(real_prob)

        # obtain anomaly score
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8942842&tag=1 1 - real_prob makes more sense
        anomaly_score = tf.add((1-lamda)*residuals, tf.cast(lamda*(1-real_prob), tf.float64))
        return anomaly_score
    
    def _replace_NaN_values(self, X: pd.DataFrame, rows_with_NaN: pd.Index) -> pd.DataFrame:
        
        mean_vals = pd.DataFrame(X.mean())

        for row in rows_with_NaN:
            X.iloc[[row]] = np.transpose(mean_vals).iloc[[0],:].values

        return X
    
    def predict_series_anomaly_scores(
            self, X: pd.DataFrame, *args, **kwargs
        ) -> pd.Series:
        
        if(X.shape[0] < self._window_size):
            # print("Not enough data to predict anomaly scores.")
            nan_array = np.empty(X.shape[0])
            nan_array[:] = np.nan
            return pd.Series(nan_array)
        
        resid = X.shape[0]%self._window_size
        
        # checking if there are any NaN values
        is_NaN = X.isnull()
        row_has_NaN = is_NaN.any(axis=1)
        rows_with_NaN = X[row_has_NaN]
        
        if(not rows_with_NaN.empty):
            X = self._replace_NaN_values(X, rows_with_NaN.index)
        
        X = self._scaler.transform(X)
        windows = [pd.DataFrame(X[i:i+self._window_size]) for i in range(0,X.shape[0]-resid,self._window_size)]
        
        anomaly_score = tf.concat([self.predict_window_anomaly_scores(X_window) for X_window in windows], -1)
        anomaly_score = pd.Series(anomaly_score).copy()
        
        if(not rows_with_NaN.empty):
            anomaly_score.at[rows_with_NaN.index] = np.nan
        
        if(resid != 0):
            # print("Ignoring last " + str(resid) + " samples (incomplete window).")
            nan_array = np.empty(resid)
            nan_array[:] = np.nan
            nan_series = pd.Series(nan_array)
            anomaly_score = pd.concat([anomaly_score, nan_series], axis=0).reset_index(drop=True) 
        
        return anomaly_score
        
    def predict_anomaly_scores(
            self, X: pd.DataFrame, *args, **kwargs
        ) -> pd.Series:
        
        if(self._id_columns == None):
            anomaly_score = self.predict_series_anomaly_scores(X)
        else:
            dfs = [x.reset_index(drop=True) for _, x in X.groupby(self._id_columns)]
            [df.drop(columns=self._id_columns, inplace=True) for df in dfs]
            anomaly_score = pd.concat([self.predict_series_anomaly_scores(df) for df in dfs], axis=0).reset_index(drop=True)
            
        return anomaly_score
    
    def identify_anomaly(self, X: pd.DataFrame, treshold: Optional[int] = 1) -> np.array:
        anomaly_score = self.predict_anomaly_scores(X)
        print("anomaly_score")
        print(anomaly_score)
        cross_entropy = np.log(anomaly_score)
        print('cross_entropy')
        print(cross_entropy)
        identified = (cross_entropy > treshold).astype(int)
        return identified