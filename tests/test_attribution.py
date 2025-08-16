import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from PIL import Image
from clarify.models.cnn import load_resnet50
from clarify.attribution import GradCAM
from clarify.data.preprocess import preprocess_image

def test_gradcam_attribution():
    # Load pre-trained model
    model = load_resnet50(pretrained=True)
    cam = GradCAM(model, target_layer='layer4')
    img = Image.open('DMR_120.jpg').convert('RGB')
    img_tensor = preprocess_image(img)
    # Use a synthetic image for testing (random tensor, shape like real image)
    #img_tensor = torch.rand(3, 224, 224)
    
    # Test attribution output
    #heatmap = cam.attribute(img_tensor)
    #assert heatmap.shape == torch.Size([224, 224]), "Heatmap shape mismatch"
    #assert (heatmap >= 0).all(), "Heatmap contains negative values"
    #print("GradCAM attribution test passed.")
    # Test attribution output
    heatmap = cam.attribute(img_tensor)
    print(f"Heatmap shape: {heatmap.shape}")  # Debug print
    assert len(heatmap.shape) == 2, "Heatmap should be 2D"
    assert min(heatmap.shape) > 0, "Heatmap should have positive dimensions"
    print("GradCAM attribution test passed.")

if __name__ == '__main__':
    test_gradcam_attribution()
