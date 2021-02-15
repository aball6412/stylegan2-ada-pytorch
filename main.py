import torch
import pickle
from training.networks import Generator, Discriminator


D = Discriminator(c_dim=0, img_resolution=32, img_channels=1)
G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=32, img_channels=1)
print(G)
print('')
print(D)


# with open('pretrained-models/metfaces.pkl', 'rb') as f:
    # G = pickle.load(f)['G_ema']  # torch.nn.Module
# z = torch.randn([1, G.z_dim]).cuda()    # latent codes
# c = None                                # class labels (not used in this example)
# img = G(z, c)  
