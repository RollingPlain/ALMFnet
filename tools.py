import torch
import cv2
import random
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F


def Get_color(save_dir, type):

    Vis_dir = 'data/test/mri_' + type + '/' + type + '/'
    F_dir = save_dir
    Save_dir = F_dir
    print(Save_dir)
    if not os.path.exists(Save_dir):
        os.makedirs(Save_dir)

    Img_names = os.listdir(F_dir)

    for name1 in Img_names:
        name = name1.split('.')[0]
        F_path = F_dir + name + '.png'
        F = cv2.imread(F_path)
        Vis_path = Vis_dir + name + '.png'
        Vis = cv2.imread(Vis_path)

        vis_YCrCb = cv2.cvtColor(Vis, cv2.COLOR_BGR2YCrCb)

        vis_YCrCb_1 = np.transpose(vis_YCrCb, (2, 0, 1))
        F_1 = np.transpose(F, (2, 0, 1))

        vis_YCrCb_1[0] = F_1[0]

        vis_YCrCb_2 = np.transpose(vis_YCrCb_1, (1, 2, 0))
        F_fin = cv2.cvtColor(vis_YCrCb_2, cv2.COLOR_YCrCb2BGR)

        save_name = Save_dir + name + '.png'
        print(save_name)
        cv2.imwrite(save_name, F_fin)
    
def data_augmentation(image, mode):
    '''
    Performs dat augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        pass
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    if random.randint(0,1) == 1:
        flag_aug = random.randint(1,7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out[0], out[1], out[2],out[3], out[4]

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


