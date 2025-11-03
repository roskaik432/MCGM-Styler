import torch
from torch.nn import functional as F
from diffusers import ModelMixin

    
class StyleEncoder(ModelMixin):
    def __init__(self):
        super().__init__()
        #self.conv1 = torch.nn.Conv2d(1, 77, kernel_size=1)
        self.lin = torch.nn.Linear(64*64*3, 1024)


    # def forward(self, styles):
    #     max_styles = torch.max(styles, axis=0).values
    #     style_inter = F.interpolate(input=max_styles, size=(64, 64))
    #     return self.lin(style_inter.view(style_inter.size(0), -1))[None]
    
    # def forward(self, styles):
    #     styles_tensor = torch.stack(styles)  # Convert the list to a Tensor
    #     #max_styles = torch.max(styles_tensor, dim=0).values #.unsqueeze(0)  # Use dim instead of axis # Add a batch dimension
    #     style_inter = F.interpolate(input=styles_tensor, size=(64, 64))
    #     return self.lin(style_inter.view(style_inter.size(0), -1))[None]

    def forward(self, styles):
        #styles_tensor = torch.stack(styles)  # Convert the list to a Tensor
        #max_styles = torch.max(styles_tensor, dim=0).values #.unsqueeze(0)  # Use dim instead of axis # Add a batch dimension
        style_inter = F.interpolate(input=styles, size=(64, 64))
        return self.lin(style_inter.view(style_inter.size(0), -1))[None]

    def merge(self, ehs1, ehs2):
        return torch.cat((ehs1, ehs2), axis=1)