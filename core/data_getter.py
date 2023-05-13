from mobility import get_survey_data
import numpy as np
import pandas as pd
from itertools import product
from keras.utils import pad_sequences
from sklearn.preprocessing import StandardScaler


class DataGetter:
    def __init__(self, n_individus: int = 1000, group_modalities: bool = True):
        self.n_individus = n_individus
        self.group_modalities = group_modalities
        self.column_id = "individual_id"
        self.columns_features = ["city_category", "csp", "n_cars"]
        self.column_pattern = "motive"
        self.column_mode = "mode_id"
        self.column_distance = "distance"
        self.pad_label = "<pad>"

        self.sample()
        if self.group_modalities:
            self.get_main_modalities()

        self.modes = self.data[self.column_mode].unique()
        self.patterns = self.data[self.column_pattern].unique()
        self.get_vocab()

    def sample(self):
        self.data = get_survey_data.get_survey_data(source="EMP-2019")["short_trips"]

        self.sample_indivs = np.random.choice(
            self.data[self.column_id].unique(), self.n_individus, replace=False
        )
        self.sample_indivs.sort()
        self.data = self.data[self.data[self.column_id].isin(self.sample_indivs)]

    def get_main_modalities(self):
        self.data[self.column_mode] = self.data[self.column_mode].str.split(
            ".", expand=True
        )[0]
        self.data[self.column_pattern] = self.data[self.column_pattern].str.split(
            ".", expand=True
        )[0]

    def get_vocab(self):
        """Associate a label for each couple (mode, pattern)"""
        self.motions = [(self.pad_label, self.pad_label)] + list(
            product(self.modes, self.patterns)
        )
        self.vocab_size = len(self.motions)
        self.vocab = {self.motions[i]: i for i in range(self.vocab_size)}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def get_features(self):
        self.features = (
            self.data[[self.column_id] + self.columns_features]
            .drop_duplicates()
            .set_index(self.column_id)
        ).sort_index()
        self.features = pd.get_dummies(self.features)
        self.n_features = self.features.shape[1]

    def get_trajectories(self):
        self.trajectories = self.data[
            [
                self.column_id,
                self.column_mode,
                self.column_pattern,
                self.column_distance,
            ]
        ]
        self.trajectories = (
            self.trajectories.groupby(self.column_id)[
                self.column_mode, self.column_pattern, self.column_distance
            ]
            .apply(lambda x: x.values.tolist())
            .sort_index()
        )

    def pad_trajectories(self):
        self.trajectory_len = max([len(seq) for seq in self.trajectories])
        self.trajectories = pad_sequences(
            self.trajectories,
            maxlen=self.trajectory_len,
            dtype=object,
            padding="post",
            truncating="post",
            value=self.pad_label,
        )
        self.trajectories[:, :, 2][self.trajectories[:, :, 2] == self.pad_label] = 0

    def distance_standardization(self):
        scaler = StandardScaler()
        self.trajectories[:, :, -1] = scaler.fit_transform(
            self.trajectories[:, :, -1].flatten().reshape(-1, 1)
        ).reshape(self.n_individus, self.trajectory_len)

        self.distance_scale = scaler.scale_
        self.distance_mean = scaler.mean_
