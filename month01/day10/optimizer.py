class SGD:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model, dW, db):

        model.W -= self.learning_rate * dW

        model.b -= self.learning_rate * db