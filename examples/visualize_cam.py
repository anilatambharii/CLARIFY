import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision
from PIL import Image
from clarify.attribution import GradCAM
from clarify.models.cnn import load_resnet50
from clarify.data.preprocess import preprocess_image
from clarify.visualizer import Visualizer

def main(img_path):
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess_image(img)

    # Load pretrained model
    model = load_resnet50(pretrained=True)

    # Initialize GradCAM and generate heatmap
    cam = GradCAM(model, target_layer='layer4')
    heatmap = cam.attribute(img_tensor)

    # Visualize using the Visualizer class
    Visualizer.show_heatmap(img_tensor, heatmap, alpha=0.5, cmap='jet')

if __name__ == "__main__":
    # Usage: python visualize_cam.py sample.jpg
    if len(sys.argv) != 2:
        print("Usage: python visualize_cam.py <path_to_image>")
    else:
        main(sys.argv[1])
