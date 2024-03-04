from pathlib import Path
import cv2
import torch
from kornia.utils import image_to_tensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from tools import random_augmentation
import os
from random import randrange

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS) 
   
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class FusionData(Dataset):  
    """
    mri, ct(?)
    """
    def __init__(self, folder: Path, tp: str, transforms=lambda x: x):
        super(FusionData, self).__init__()
        img_names = [x.name for x in sorted(folder.glob('mri/*')) if x.suffix in ['.bmp', '.png', '.jpg']]
        self.samples = [{'name': x, 'mri': folder / 'mri' / x, 'oth': folder / tp / x} for x in img_names]
        self.transforms = transforms
        self.type = tp

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        # source
        mri = cv2.imread(str(sample['mri']))
        oth = cv2.imread(str(sample['oth']))
        mri = cv2.cvtColor(mri, cv2.COLOR_BGR2YCrCb)[:,:,0]
        oth = cv2.cvtColor(oth, cv2.COLOR_BGR2YCrCb)[:,:,0]
    
        im_list = [image_to_tensor(x) / 255. for x in [mri, oth]]
        im = torch.cat(im_list, dim=0)
        mri, oth = torch.chunk(self.transforms(im), 2, dim=0)  
        return mri, oth, sample['name']


class TrainingData(Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/mri')
        self.dir_B = os.path.join(self.root, self.phase + '/oth')
        self.dir_C = os.path.join(self.root, self.phase + '/mri_mask')
        self.dir_D = os.path.join(self.root, self.phase + '/oth_mask')
        self.dir_E = os.path.join(self.root, self.phase + '/seg')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.D_paths = sorted(make_dataset(self.dir_D))
        self.E_paths = sorted(make_dataset(self.dir_E))

        # transform
        self.transform = ToTensor()  

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("L")
        B = Image.open(self.B_paths[index]).convert("L")
        C = Image.open(self.C_paths[index]).convert('L')
        D = Image.open(self.D_paths[index]).convert("L")
        E = Image.open(self.E_paths[index]).convert("L")

        w, h = A.size
        x, y = randrange(w - self.fineSize + 1), randrange(h - self.fineSize + 1)
        cropped_a = A.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_b = B.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_c = C.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_d = D.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_e = E.crop((x, y, x + self.fineSize, y + self.fineSize))

        im_ir, im_vis, im_mask_ir, im_mask_vis, seg = random_augmentation(cropped_a,cropped_b,cropped_c,cropped_d,cropped_e)

        tensor_1 = self.transform(im_ir)
        tensor_2 = self.transform(im_vis)
        tensor_3 = self.transform(im_mask_ir)
        tensor_4 = self.transform(im_mask_vis)
        tensor_5 = self.transform(seg)

        im_list = [tensor_1,tensor_2,tensor_3,tensor_4,tensor_5]

        return torch.cat(im_list, dim=0)

    def __len__(self):
        return len(self.A_paths)


