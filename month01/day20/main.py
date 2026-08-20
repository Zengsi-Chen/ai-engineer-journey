import torch

from model import BinaryClassifier
from inference import ClassificationInference


# 1. 创建模型
model = BinaryClassifier()


# 2. 加载 Best Model
model.load_state_dict(
    torch.load(
        "../day18/best_model.pth",
        map_location="cpu"
    )
)


# 3. Day 18 找到的最佳 threshold
best_threshold = 0.5


# 4. 创建 inference pipeline
classifier = ClassificationInference(
    model=model,
    threshold=best_threshold
)


# 5. 新数据
x = torch.tensor([
    [2.5],
    [5.0],
    [7.5],
    [10.0]
])


# 6. Prediction
result = classifier.predict(x)


print("Input:", x)
print("Probability:", result.probability)
print("Prediction:", result.prediction)