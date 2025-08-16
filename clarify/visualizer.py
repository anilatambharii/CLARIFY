import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

class Visualizer:
    @staticmethod
    def show_heatmap(img_tensor, heatmap, alpha=0.5, cmap='jet'):
        img = ToPILImage()(img_tensor)
        plt.imshow(img)
        plt.imshow(heatmap, alpha=alpha, cmap=cmap)
        plt.colorbar()
        plt.axis('off')
        plt.show()
