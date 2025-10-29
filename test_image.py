import torch
from PIL import Image
from torchvision import transforms, models
import json
import os

#Load class names
if not os.path.exists('class_names.json'):
    print("ERROR: class_names.json not found!")
    exit()

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

#Load model ARCHITECTURE & WEIGHTS 
if not os.path.exists('models/plant_disease_resnet50.pth'):
    print("ERROR: Model not found!")
    exit()

model = models.resnet50(pretrained=False)
num_classes = len(class_names)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load('models/plant_disease_resnet50.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

#Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Predict function
def predict_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        pred_idx = torch.argmax(output, dim=1).item()
    return class_names[pred_idx]

#Test
if __name__ == "__main__":
    test_img = "test_images/sample_leaf.jpg"
    if os.path.exists(test_img):
        disease = predict_disease(test_img)
        print(f"\nPREDICTED DISEASE: {disease}\n")
    else:
        print(f"Image not found: {test_img}")
        print("Put a leaf image in 'test_images/sample_leaf.jpg'")