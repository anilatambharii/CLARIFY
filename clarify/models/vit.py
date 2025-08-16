try:
    import timm
except ImportError:
    raise ImportError("timm library is required to load Vision Transformer models. Install with 'pip install timm'.")

def load_vit_model(model_name='vit_base_patch16_224', pretrained=True):
    """
    Load a pretrained Vision Transformer model using timm.
    """
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    return model
