import sys
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import pickle
from genotypes import PRIMITIVES

sys.path.append('..')

from tools import measure_latency_in_ms
from operations import OPS
cudnn.enabled = True
cudnn.benchmark = True


def get_latency_lookup_en(is_cuda):
    latency_lookup = OrderedDict()

    for type, C in zip(['cell_f1', 'cell_f2', 'cell_f'], [16, 16, 16]):
        latency_lookup[type] = OrderedDict()
        for j in range(len(PRIMITIVES)):
            op = OPS[PRIMITIVES[j]](C)
            shape = [1, C, 128, 128]
            print(shape)
            lat = measure_latency_in_ms(op, shape, is_cuda)
            latency_lookup[type][PRIMITIVES[j]] = lat

    print(latency_lookup)
    return latency_lookup

if __name__ == '__main__':
    print('measure latency on gpu......')
    latency_lookup = get_latency_lookup_en(is_cuda=True)
    with open('latency_noten.pkl', 'wb') as f:
        pickle.dump(latency_lookup, f)