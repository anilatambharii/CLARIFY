import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from PIL import Image
from clarify.visualizer import Visualizer
from clarify.data.preprocess import preprocess_image  # Add this import

def test_show_heatmap():
    # Create a dummy image and heatmap
    img = Image.open('DMR_120.jpg').convert('RGB')
    img_tensor = preprocess_image(img)
    # Alternatively, use a random tensor:
    # img_tensor = torch.rand(3, 224, 224)
    heatmap = torch.rand(224, 224)
    
    # This should display the visualization without errors
    try:
        Visualizer.show_heatmap(img_tensor, heatmap)
        print("Visualizer test passed.")
    except Exception as e:
        print(f"Visualizer test failed: {e}")

if __name__ == '__main__':
    test_show_heatmap()
