from dataset import create_dataloaders
from cnn_model import SimpleCNN


train_loader, _ = create_dataloaders()

images, labels = next(iter(train_loader))

model = SimpleCNN()

output = model(images)

print("Images :", images.shape)
print("Labels :", labels.shape)
print("Output :", output.shape)

prediction = output.argmax(dim=1)

print("Predictions:", prediction[:10])
print("Labels     :", labels[:10])