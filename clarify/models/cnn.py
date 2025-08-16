import torchvision.models as models

def load_resnet50(pretrained=True):
    """
    Load the pretrained ResNet50 model.
    """
    model = models.resnet50(pretrained=pretrained)
    model.eval()
    return model
