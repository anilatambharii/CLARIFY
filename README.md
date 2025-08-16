# CLARIFY: Open Source Toolkit for Interpretable Computer Vision
CLARIFY provides practical tools for feature attribution and visualization for CNNs and Vision Transformers.

**CLARIFY** is a Python framework designed to provide fast, modular, and production-ready tools for explainable computer vision models. It supports a variety of attribution methods for CNNs and Vision Transformers, with easy-to-use visualization and benchmarking functionalities.

## Features

- Feature attribution methods including Grad-CAM, Integrated Gradients, and Attention Rollout  
- Support for CNN and Vision Transformer architectures  
- Interactive and scriptable visualization modules  
- Modular design for easy integration and extension  
- Example notebooks for quick experimentation and learning  

## Installation

git clone https://github.com/anilatambharii/CLARIFY.git
cd CLARIFY
pip install -r requirements.txt

## Quick Start
Load a pretrained ResNet50, generate Grad-CAM heatmaps, and visualize:
from clarify.attribution import GradCAM
from clarify.models.cnn import load_resnet50
import torchvision

model = load_resnet50(pretrained=True)
img = torchvision.io.read_image('sample.jpg')
cam = GradCAM(model)
heatmap = cam.attribute(img)
cam.visualize(img, heatmap)

## Usage Examples

Check the `examples/` folder for Jupyter notebooks and scripts demonstrating how to use various attribution methods and visualizers.

## Contribution

We encourage contributions from the community! Ways you can help:  
- Add support for new architectures (e.g., EfficientNet, Swin Transformer)  
- Implement additional interpretability algorithms  
- Optimize attribution for resource-constrained or edge devices  
- Improve documentation and tutorials  
- Develop domain-specific extensions (healthcare, satellite imagery, etc.)  

Please read our **CONTRIBUTING.md** file for guidelines.
## License

CLARIFY is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contact

Feel free to open issues or reach out via GitHub Discussions for questions, feedback, or collaborations.

---
