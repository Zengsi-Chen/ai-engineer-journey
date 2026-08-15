class LinearGradient:

    def backward(self, X, prediction, target):

        n = len(X)

        error = prediction - target

        dW = (2 / n) * (X.T @ error)

        db = (2 / n) * error.sum()

        return dW, db