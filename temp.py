import actrix
model = actrix.model("mvd_vit_large", num_classes=10, img_size=(64,128))

print(model.config)

print(model)

import torch
x = torch.rand(size=(2,3,16,128,64))

out = model(x)

print(out.shape)