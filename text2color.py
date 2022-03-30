import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os


class Text2Color(nn.Module):
    # Same base then split into two separate modules
    def __init__(self, input_dim=512, width=512, out_dim=512, mlp_depth=2):
        super(Text2Color, self).__init__()
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(mlp_depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, out_dim))
        self.mlp = nn.ModuleList(layers)

    def forward(self, input):
        z = input.float()
        for layer in self.mlp:
            z = layer(z)
        return z


def save_model(model, loss, iter, optim, output_dir):
    save_dict = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss
    }
    path = os.path.join(output_dir, 'checkpoint.pth.tar')
    torch.save(save_dict, path)
