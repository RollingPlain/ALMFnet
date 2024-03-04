from pathlib import Path
import cv2
import torch
from kornia import tensor_to_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from tqdm import tqdm
from dataset import FusionData
from pathlib import Path
from kornia.utils import tensor_to_image
from model_DAG import Network_DAG as Network
import genotypes
from tools import Get_color
import argparse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data/test/mri_ct/', help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./cache_train/ALMFNet.pt', help='location of the checkpoint')
parser.add_argument('--save_dir', type=str, default='./runs/mri_ct/', help='location to save images')
parser.add_argument('--id', type=int, default=0, help='location of the data corpus')
parser.add_argument('--mode', type=str, default='ct', help='ct, pet, spect')
parser.add_argument('--name', type=str, default='DAG_lat_', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--img_size', type=int, default=256, help='image size')
parser.add_argument('--arch', type=str, default='DAG_lat', help='which architecture to use')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
args = parser.parse_args()

def eval_process(cp: Path, folder: Path, dst: Path, type):

    torch.backends.cudnn.benchmark = True
    dst.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.gpu)

    # dataset
    to_float = transforms.ConvertImageDtype(torch.float)
    dataset = FusionData(folder, type, to_float)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    genotype = eval("genotypes.%s" % args.arch)
    net = Network(args.init_channels, genotype)  
    if torch.cuda.is_available():
        net.cuda()


    net.load_state_dict(torch.load(cp))
    net.eval()

    process = tqdm(enumerate(dataloader))
    for batch_idx, (mri, oth, name) in process:
        process.set_description(f'fusing image: {name})')

        if torch.cuda.is_available():
            im = torch.cat([mri, oth], dim=1) 
            im = im.cuda()
            mri, oth = torch.chunk(im, 2, dim=1)

        with torch.no_grad():

            total_fusion = sum([param.nelement() for param in net.parameters()])
            fus = net(mri, oth)

        torchvision.utils.save_image(fus[0], str(dst / name[0]))
        cv2.imwrite(str(dst / name[0]), tensor_to_image(fus[0] * 255.))


if __name__ == '__main__':

    checkpoint = Path(args.checkpoint)
    input_path = args.data
    dataset_folder = Path(input_path)
    dst = Path(args.save_dir)
    
    eval_process(checkpoint, dataset_folder, dst, args.mode)
    
    Get_color(args.save_dir, args.mode)


    
        
