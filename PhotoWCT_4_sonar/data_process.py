import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import cv2 as cv

data_dir = './data/'
save_dir = './data/'
folder_name_list = os.listdir(data_dir)

for folder_name in folder_name_list:

    print(folder_name)
    if folder_name == 'other':
        print('debug')

    if os.path.exists(data_dir+folder_name):

        file_name_list = os.listdir(data_dir+folder_name)

        for i in range(len(file_name_list)):


            name = file_name_list[i]
            print(name)

            img = cv.imread(data_dir+folder_name+'/'+name)

            # if len(img.shape)>2:
            #     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            img = cv.resize(img,(256,256))

            if not os.path.exists(save_dir+folder_name):
                os.mkdir(save_dir+folder_name)

            cv.imwrite(save_dir+folder_name+'/'+ folder_name + str(i)+'.jpg', img)