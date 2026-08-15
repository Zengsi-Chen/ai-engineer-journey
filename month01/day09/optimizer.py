class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model, dW, db):
        model.W = model.W - self.learning_rate * dW
        model.b = model.b - self.learning_rate * db


if __name__ == "__main__":
    from model import LinearModel

    model = LinearModel()

    optimizer = SGD(learning_rate=0.1)

    dW = 2.0
    db = 1.0

    print("Before update:")
    print("W:", model.W)
    print("b:", model.b)

    optimizer.step(model, dW, db)

    print("After update:")
    print("W:", model.W)
    print("b:", model.b)