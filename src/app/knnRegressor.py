from sklearn.neighbors import KNeighborsRegressor

import logging

logging.getLogger().setLevel(logging.INFO)


class KNNRegression(object):
    """Very simple KNNRegression class to try to predict the target values.

    The absolute minimum implementation.
    """

    def __init__(self,
                 train_x,
                 train_y,
                 test_x,
                 test_y,
                 n_neighbours):
        """

        :param x_columns:
        :param y_column:
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.n_neighbours = n_neighbours

    def run(self):
        """Function to run process of KNNRegressor.
        A nice thing is that sklearn performs normalization for us (nice! =D) and distance finding automatically.

        :return: predictions, actual_target_values, mse (mean squared error)
        """
        knnr = KNeighborsRegressor(n_neighbors=self.n_neighbours)
        knnr.fit(self.train_x, self.train_y)
        predictions = knnr.predict(self.test_x)

        # Get the actual values for the test set.
        actual_target_values = self.test_y

        # Compute the mean squared error of predictions
        mse_pred = (((predictions - actual_target_values) ** 2).sum()) / len(predictions)

        return [predictions, actual_target_values, mse_pred]
