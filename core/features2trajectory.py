import numpy as np
from .data_getter import DataGetter
from .mob2vec import Mob2Vec
from keras.models import Sequential
from keras.layers import (
    LSTM,
    Dense,
    InputLayer,
)
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix


class Features2Trajectory:
    def __init__(self, dg: DataGetter, m2v: Mob2Vec, n_lstm_units: int = 10):
        self.dg = dg
        self.m2v = m2v

        self.model = Sequential()
        self.model.add(
            InputLayer(input_shape=(self.dg.trajectory_len, self.dg.n_features))
        )
        self.model.add(LSTM(n_lstm_units, return_sequences=True, activation="relu"))
        self.model.add(Dense(self.m2v.embedding_dim + 1))

    def get_trajectories_for_rnn(self):
        self.trajectories_data = np.zeros(
            (self.dg.n_individus, self.dg.trajectory_len, self.m2v.embedding_dim + 1)
        )
        self.trajectories_data[:, :, -1] = self.dg.trajectories[:, :, -1]
        self.trajectories_data[
            :, :, : self.m2v.embedding_dim
        ] = self.m2v.embedding_layer.embeddings.numpy()[
            self.m2v.trajectories_encoded.flatten().astype(int), :
        ].reshape(
            (self.dg.n_individus, self.dg.trajectory_len, self.m2v.embedding_dim)
        )

    def pad_features(self, features):
        features_padded = np.pad(
            features[:, np.newaxis, :],
            ((0, 0), (0, self.dg.trajectory_len - 1), (0, 0)),
        )
        return features_padded

    def get_feature_for_rnn(self):
        features = self.dg.features.values
        self.features_padded = self.pad_features(features)

    def fit(self, test_size=0.33, batch_size=100, epochs=20):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features_padded, self.trajectories_data, test_size=test_size
        )

        self.model.compile(optimizer="rmsprop", loss="mse")
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            batch_size=batch_size,
            epochs=epochs,
        )

    def predict(self, features_padded):
        trajectories_pred = self.model.predict(features_padded)
        return trajectories_pred

    def get_closest_embedding(self, trajectory_pred):
        motion_ids = distance_matrix(
            self.m2v.embedding_layer.embeddings.numpy(),
            trajectory_pred[0, :, : self.m2v.embedding_dim],
        ).argmin(0)
        return motion_ids

    def readable_prediction(self, trajectory_pred):
        trajectory_pred[0, :, -1] = (
            trajectory_pred[0, :, -1] * self.dg.distance_scale + self.dg.distance_mean
        )
        motion_ids = self.get_closest_embedding(trajectory_pred)
        trajectory_pred_lib = [
            self.dg.inverse_vocab[motion_ids[i]] + (trajectory_pred[0, i, -1],)
            for i in range(self.dg.trajectory_len)
        ]
        return trajectory_pred_lib

    def readeable_true(self, indiv):
        trajectory_true_lib = [
            self.dg.inverse_vocab[
                np.where(
                    (
                        self.m2v.embedding_layer.embeddings.numpy()
                        == self.y_test[indiv, i, : self.m2v.embedding_dim]
                    ).all(1)
                )[0][0]
            ]
            + (
                self.y_test[indiv, i, -1] * self.dg.distance_scale[0]
                + self.dg.distance_mean[0],
            )
            for i in range(self.dg.trajectory_len)
        ]
        return trajectory_true_lib

    def compare_pred_true(self, indiv):
        trajectory_pred = self.model.predict(self.X_test[[indiv]])
        trajectory_pred_lib = self.readable_prediction(trajectory_pred)
        trajectory_true_lib = self.readeable_true(indiv)
        return trajectory_pred_lib, trajectory_true_lib
