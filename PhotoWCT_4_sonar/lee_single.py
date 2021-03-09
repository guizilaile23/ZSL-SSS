#################################################################################
#
#
#
#################################################################################

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils

import os

from networks import Sonar_noise_WCT
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image

###############################################################################################
###############################################################################################

content_image_path = './data/content/9-roundabout_train_53.jpg'
style_image_path = './data/style/'
img_n = '3-'

output_image_path = './data/output/'
if not os.path.exists(output_image_path):
    os.mkdir(output_image_path)

###############################################################################################

# Load model
model_name = './PhotoWCTModels/photo_wct.pth'

sn_wct = Sonar_noise_WCT()
sn_wct.load_state_dict(torch.load(model_name))

sn_wct.cuda()

###############################################################################################################################################
#######     Stylization  ######################################################################################################################
###############################################################################################################################################

cont_img = Image.open(content_image_path).convert('RGB')
cont_img = cont_img.resize((256, 256))
cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
cont_img = cont_img.cuda()

style_image_list = os.listdir(style_image_path)
style_image_list.sort()
num_style = len(style_image_list)

for i in range(num_style):

    style_name = style_image_list[i]
    styl_img = Image.open(style_image_path + style_name).convert('RGB')
    styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)

    print(i, '  ', style_name)
    with torch.no_grad():

        styl_img = styl_img.cuda()

        stylized_img = sn_wct.transform(cont_img, styl_img, 1.0)

        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        #### save img

        s_name = 'NoiseWCT_-'+ img_n + style_name.split('.')[0] +'-1.0.jpg'
        save_name = os.path.join(output_image_path, s_name)

        out_img.save(save_name)
        out_img.show()

#####################################################################
