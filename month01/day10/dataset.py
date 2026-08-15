import numpy as np


class Dataset:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class DataLoader:

    def __init__(self, dataset, batch_size=4, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):

        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(
            0,
            len(indices),
            self.batch_size
        ):

            batch_indices = indices[
                start:start + self.batch_size
            ]

            X_batch = self.dataset.X[batch_indices]
            y_batch = self.dataset.y[batch_indices]

            yield X_batch, y_batch

