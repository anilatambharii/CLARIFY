from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def preprocess_image(img, image_size=224):
    """
    Preprocess PIL image for model input.
    """
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)
