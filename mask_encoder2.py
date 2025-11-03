import torch
from torch.nn import functional as F
from diffusers import ModelMixin

# class MaskEncoder(ModelMixin):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(1, 77, kernel_size=1)
#         self.lin = torch.nn.Linear(64*64, 1024)
#         self.merger = torch.nn.Conv1d(154, 77, kernel_size=1)

#     def forward(self, mask):
#         max_masks = torch.max(mask, axis=1).values
#         mask = F.interpolate(
#             input=max_masks, size=(64, 64)
#         )
#         mask = self.conv1(mask)
#         mask = self.lin(mask.view(mask.shape[:2] + (-1,)))
#         return mask
    
#     def merge(self, ehs1, ehs2):
#         print ("the ehs1 shape : " ,ehs1.shape)
#         print ("the ehs2 shape : " ,ehs2.shape)
#         return self.merger(torch.cat((ehs1, ehs2), axis=1))
    
class MaskEncoder(ModelMixin):
    def __init__(self):
        super().__init__()
        #self.conv1 = torch.nn.Conv2d(1, 77, kernel_size=1)
        self.lin = torch.nn.Linear(64*64, 1024)
        # self.lin2 = torch.nn.Linear(1024, 512)
        # self.lin3 = torch.nn.Linear(512, 256)
        # self.output_layer = torch.nn.Linear(256, 77)
        #self.merger = torch.nn.Conv1d(154, 77, kernel_size=1)

    # def forward(self, mask):
    #     max_masks = torch.max(mask, axis=1).values
    #     mask = F.interpolate(
    #         input=max_masks, size=(64, 64)
    #     )
    #     mask = self.lin(mask.view(mask.size(0),-1))       
    #     return mask.unsqueeze(1)
    
    # def merge(self, ehs1, ehs2):
    #     print ("the ehs1 shape : " ,ehs1.shape)
    #     print ("the ehs2 shape : " ,ehs2.shape)
    #     return torch.cat((ehs1, ehs2), axis=1)

    def forward(self, masks):
        #max_masks = torch.max(masks, axis=0).values
        #mask_inter = F.interpolate(input=masks[0], size=(64, 64))
        max_masks = torch.max(masks, axis=0).values
        mask_inter = F.interpolate(input=max_masks, size=(64, 64))
        return self.lin(mask_inter.view(mask_inter.size(0), -1))[None]
        # encoded_masks = []
        # for mask in masks[0]:
        #     max_masks = torch.max(mask, axis=0).values
        #     mask_inter = F.interpolate(input=max_masks[None], size=(64, 64))
        #     mask_encoded = self.lin(mask_inter.view(mask_inter.size(0), -1))
        #     encoded_masks.append(mask_encoded.unsqueeze(1))
        # return torch.cat(encoded_masks, dim=1)

    def merge(self, ehs1, ehs2):
        return torch.cat((ehs1, ehs2), axis=1)