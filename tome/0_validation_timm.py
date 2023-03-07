import timm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# 随便找一个已有的vit 预训练好的模型：
model_name = 'vit_base_patch16_224'
# NOTE 86,567,656 = 86.6M learnable parameters
model = timm.create_model(model_name, pretrained=True)

input_size = model.default_cfg['input_size'][1] # 224

transform = transforms.Compose([
    transforms.Resize(int((256/224)*input_size), 
        interpolation=InterpolationMode.BICUBIC), 
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(model.default_cfg['mean'], # (0.5, 0.5, 0.5)
        model.default_cfg['std']) # (0.5, 0.5, 0.5)
])

img = Image.open('images/husky.png')
img_tensor = transform(img)[None, ...] # (1, 3, 224, 224) 这是经过变形之后的一张图片的张量

# 这是使用已有的pretrained vit模型，来预测一张图片:
outlist = model(img_tensor).topk(5).indices[0].tolist()
print(outlist)

# [248, 250, 249, 269, 273], top-5的预测结果

# apply_patch
from patch.timm import apply_patch

# NOTE this is the important part!
import ipdb; ipdb.set_trace()
apply_patch(model)

import ipdb; ipdb.set_trace()
model.r = 4
outlist2 = model(img_tensor).topk(5).indices[0].tolist()
print('model.r={}, prediction={}'.format(model.r, outlist2))
# [248, 250, 249, 269, 273]

import ipdb; ipdb.set_trace()
model.r = 8
outlist3 = model(img_tensor).topk(5).indices[0].tolist()
print('model.r={}, prediction={}'.format(model.r, outlist3))
# [248, 250, 249, 269, 537 NOTE the final prediction changed!]

import ipdb; ipdb.set_trace()
model.r = 16
outlist4 = model(img_tensor).topk(5).indices[0].tolist()
print('model.r={}, prediction={}'.format(model.r, outlist4))

