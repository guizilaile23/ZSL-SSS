
from __future__ import print_function
import torch

from networks import Sonar_noise_WCT

import time
import numpy as np
from PIL import Image

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch

import os
import random
import matplotlib.pyplot as plt
import time
###############################################################################################
def imshow(inp, title=None):
    """Imshow for Tensor."""


    inp = inp.cpu().numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

###############################################################################################

content_dir = '../000-Data/content_gray/'
style_dir = '../000-Data/style/gray/'
save_dir = '../000-Data/output-3-PhotoWCT_sonar-gray-'

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    #transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

Content_set = torchvision.datasets.ImageFolder(content_dir, transform)
Style_set = torchvision.datasets.ImageFolder(style_dir, transform)

###############################################################################################

# Load model
model_name = './weights/weight.pth'

p_wct = Sonar_noise_WCT()
p_wct.load_state_dict(torch.load(model_name))

p_wct.cuda(0)



###############################################################################################################################################
#######     Stylization  ######################################################################################################################
###############################################################################################################################################

style_number = 2

cuda = True

number_of_samples = len(Content_set)
print('total', number_of_samples)
number_of_styles = len(Style_set)
print('style number :', len(Style_set))

with torch.no_grad():

    for i in range(0, number_of_samples):

        start_time = time.time()
        cont_img = Content_set[i][0].unsqueeze(0)
        cont_img = cont_img.cuda()

        cont_dir = Content_set.samples[i][0]
        cont_label = Content_set.classes[Content_set[i][1]]
        print('total', number_of_samples, 'current:', i, ':', cont_dir)

        style_idx = random.sample(range(len(Style_set)), style_number)
        for j in style_idx:

            styl_img = Style_set[j][0].unsqueeze(0)
            styl_img = styl_img.cuda()

            for k in range(1,11):
                yita = 0.1*k
                stylized_img = p_wct.transform(cont_img, styl_img, yita)


                if not os.path.exists(save_dir+str(k)):
                    os.mkdir(save_dir+str(k))
                if not os.path.exists(os.path.join(save_dir+str(k), cont_label)):
                    os.mkdir(os.path.join(save_dir+str(k), cont_label))
                #### save img
                save_name = os.path.join(save_dir+str(k), cont_label, str(i) + '-' + str(j) + '.jpg')
                torchvision.utils.save_image(stylized_img.cpu().detach().squeeze(0), save_name)

        img_time = time.time() - start_time
        print('total time cost: ', img_time)
            # grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
            # ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            # out_img = Image.fromarray(ndarr)
            # out_img.save(save_name)
            # out_img.show()



