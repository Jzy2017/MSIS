from collections import namedtuple

Genotype = namedtuple('Genotype', 'recon_en recon_de recon_en2 recon_de2')


DARTS_stitch = Genotype(
                recon_en=[('dil_conv_5x5', 0), ('conv_1x1', 0), ('dil_conv_5x5', 1), ('conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3), ('conv_1x1', 2), ('conv_1x1', 2), ('conv_3x3', 1)],
                recon_de=[('conv_5x5', 0), ('conv_1x1', 1), ('dil_conv_5x5', 0), ('conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('conv_1x1', 3)], 
                recon_en2=[('dil_conv_5x5', 0), ('conv_3x3', 0), ('dil_conv_5x5', 1), ('conv_3x3', 2), ('conv_5x5', 0), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_1x1', 4), ('conv_5x5', 2)], 
                recon_de2=[('conv_1x1', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('conv_1x1', 2), ('conv_1x1', 2), ('conv_3x3', 0)])