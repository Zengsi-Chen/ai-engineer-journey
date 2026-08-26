from dataset import create_dataloaders
import matplotlib.pyplot as plt


train_loader, test_loader = create_dataloaders()

images, labels = next(iter(train_loader))

print("Batch shape:", images.shape)
print("Labels shape:", labels.shape)

image = images[0]

print("Single image shape:", image.shape)
print("Label:", labels[0].item())

print("Pixel [10, 15]:", images[0, 0, 10, 15].item())

plt.imshow(images[0, 0], cmap="gray")
plt.title(f"Label: {labels[0].item()}")
plt.axis("off")
plt.show()