from collections import namedtuple
Genotype = namedtuple('Genotype', 'f1 f1_cat f2 f2_cat f f_cat')


PRIMITIVES = [
    'den_conv_1x1',
    'den_conv_3x3',
    'den_conv_5x5',
    'res_conv_1x1',
    'res_conv_3x3',
    'res_conv_5x5',
    'sep_conv_1x1',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_3x1_1x3',
    'conv_5x1_1x5',
    'ECAattention',
    'SPAattention'
]


DAG_lat = Genotype(f1=[('res_conv_3x3', 0), ('conv_5x1_1x5', 0), ('sep_conv_7x7', 1), ('den_conv_1x1', 0), ('conv_3x1_1x3', 1), ('SPAattention', 0), ('sep_conv_5x5', 1)], f1_cat=range(1, 5), f2=[('sep_conv_1x1', 0), ('conv_3x1_1x3', 1), ('sep_conv_7x7', 0), ('conv_3x1_1x3', 0), ('sep_conv_7x7', 1), ('sep_conv_3x3', 2), ('res_conv_3x3', 0)], f2_cat=range(1, 5), f=[('sep_conv_1x1', 0), ('conv_3x1_1x3', 1), ('sep_conv_1x1', 0), ('sep_conv_1x1', 2), ('sep_conv_1x1', 1), ('den_conv_1x1', 0), ('sep_conv_7x7', 3)], f_cat=range(1, 5))

