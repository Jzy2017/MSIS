from collections import namedtuple
Genotype = namedtuple('Genotype', 'ir vis decoder')
DARTS_fusion = Genotype(
                        ir=[('conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('conv_1x1', 1), ('dil_conv_5x5', 2), ('conv_1x1', 2), ('conv_3x3', 1)], 
                        vis=[('dil_conv_5x5', 0), ('conv_3x3', 1), ('conv_5x5', 0), ('conv_1x1', 2), ('conv_3x3', 1), ('conv_3x3', 2), ('conv_1x1', 0)], 
                        decoder=[('conv_1x1', 0), ('dil_conv_5x5', 0), ('conv_3x3', 1), ('conv_1x1', 2), ('dil_conv_3x3', 1), ('conv_1x1', 1), ('conv_1x1', 0)]
                        )

