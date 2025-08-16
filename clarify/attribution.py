import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.target_layer = target_layer or list(model._modules.items())[-1][0]
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def attribute(self, img_tensor, class_idx=None):
        output = self.model(img_tensor.unsqueeze(0))
        class_idx = class_idx if class_idx is not None else output.argmax()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()
        pooled_grad = torch.mean(self.gradients, dim=[0, 2, 3])
        weighted_activations = self.activations * pooled_grad.unsqueeze(-1).unsqueeze(-1)
        heatmap = weighted_activations.mean(dim=1).squeeze().cpu().detach()
        return F.relu(heatmap)
    
    def visualize(self, img_tensor, heatmap):
        import matplotlib.pyplot as plt
        from torchvision.transforms import ToPILImage

        img = ToPILImage()(img_tensor)
        plt.imshow(img)
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.colorbar()
        plt.show()
