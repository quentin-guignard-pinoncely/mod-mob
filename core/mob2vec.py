import numpy as np
from .data_getter import DataGetter
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (
    Dense,
    GlobalAveragePooling1D,
    Embedding,
)


class Mob2Vec:
    def __init__(self, dg: DataGetter, window_size: int = 2, embedding_dim: int = 2):
        self.dg = dg
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.encoding()
        self.generate_data_cbow()

        self.embedding_layer = Embedding(
            input_dim=self.dg.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.window_size * 2,
            embeddings_initializer="RandomNormal",
        )
        self.model = Sequential(
            [
                self.embedding_layer,
                GlobalAveragePooling1D(),
                Dense(self.dg.vocab_size, activation="softmax"),
            ]
        )

    def encoding(self):
        self.trajectories_encoded = np.zeros(
            (self.dg.n_individus, self.dg.trajectory_len)
        )
        for motion, index in self.dg.vocab.items():
            self.trajectories_encoded[
                (self.dg.trajectories[:, :, :2] == motion).all(axis=2)
            ] = index

        self.trajectories_encoded = self.trajectories_encoded.astype(int)

    def generate_data_cbow(self):
        """
        generate all couple (context, target) from the trajectories.
        Exemple of one couple (sequence, context) :
            for a sequence [5,2,3,4,2]:
                context = [5,2,,4,2], target = 3
        Then the target is one-hot encoded
        """
        self.contexts = []
        self.targets = []
        for target in range(self.dg.trajectory_len):
            full_context = np.delete(self.trajectories_encoded, target, 1)
            left_index = np.maximum(target - self.window_size, 0)
            left_padding_size = left_index - (target - self.window_size)
            right_index = np.minimum(
                target + self.window_size, self.dg.trajectory_len - 1
            )
            right_padding_size = (target + self.window_size) - right_index

            context = full_context[:, left_index:right_index]
            context = np.pad(context, ((0, 0), (left_padding_size, right_padding_size)))
            self.contexts.append(context)

            target_one_hot_encoded = np.zeros(
                (self.dg.n_individus, self.dg.vocab_size), dtype=int
            )
            target_one_hot_encoded[
                range(self.dg.n_individus), self.trajectories_encoded[:, target]
            ] = 1
            self.targets.append(target_one_hot_encoded)

        self.contexts = np.concatenate(self.contexts, 0)
        self.targets = np.concatenate(self.targets, 0)

        # drop only pad rows
        to_keep = self.contexts.sum(1) > 0
        self.contexts = self.contexts[to_keep]
        self.targets = self.targets[to_keep]

    def fit(self, test_size=0.1, batch_size=500, epochs=20):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.contexts, self.targets, test_size=test_size
        )

        self.model.compile(optimizer="adam", loss="categorical_crossentropy")
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test),
        )
