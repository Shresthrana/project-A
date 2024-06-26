import torch
from PIL import Image
import torchvision
from torchvision import models
from torchvision.models import ResNet101_Weights
from torchvision import transforms

# Correct way to load the model with weights
model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
model.eval()

# Image path (ensure this path is correct)
# image_path = r'C:\Users\a_k_r\Desktop\collegematerial\imagerec\animals\animals\bat\1dd514de63.jpg'
image_path = r"C:\Users\a_k_r\Desktop\collegematerial\imagerec\animals\animals\ox\0e1892d5e1.jpg"
#
# # Open the image
try:
    image = Image.open(image_path)
except FileNotFoundError:
    print(f"File not found: {image_path}")
    exit()
#
# # Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),      # Scale the input image to 256x256
    transforms.CenterCrop(224),  # Crop the image to 224x224
    transforms.ToTensor(),       # Convert the image to a tensor
    transforms.Normalize(        # Normalize with mean and std deviation
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
#
# # Apply the transformations
image_t = preprocess(image)

# Create a mini-batch as expected by the model
batch_t = torch.unsqueeze(image_t, 0)
#
# # Make sure the model is in evaluation mode
model.eval()
#
# # Forward pass through the model
with torch.no_grad():  # Disable gradient calculation
    out = model(batch_t)
#
# # Read the labels from the file
with open("imagenet target.txt") as f:
    labels = [line.strip() for line in f.readlines()]
#
# # Get the predicted class
# print(out)
_, index = torch.max(out, 1)

#
# # Convert output probabilities to percentages
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# Print the label and confidence
print(f"Predicted: {labels[index[0]]}, Confidence: {percentage[index[0]].item():.2f}%")
