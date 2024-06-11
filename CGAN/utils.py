import os
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt

# custom weights initialization
# Reference (PyTorch Tutorials)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_images(images, output_dir):
    print('Saving Images...')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for idx, img in enumerate(images):
        img = img.numpy()
        c, h, w = img.shape
        img = img.reshape(h, w, c)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'image_{timestamp}_{idx}.png'
        filepath = os.path.join(output_dir, filename)
        plt.imsave(filepath, img)
    print("Done!")
    return True