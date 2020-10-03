from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]
STEPS = 8
CONCAT = 8

ENAS = Genotype(
    recurrent = [
        ('tanh', 0),
        ('tanh', 1),
        ('relu', 1),
        ('tanh', 3),
        ('tanh', 3),
        ('relu', 3),
        ('relu', 4),
        ('relu', 7),
        ('relu', 8),
        ('relu', 8),
        ('relu', 8),
    ],
    concat = [2, 5, 6, 9, 10, 11]
)

DARTS_V1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4), ('identity', 1), ('relu', 5), ('relu', 1)], concat=range(1, 9))
DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))

DARTS = DARTS_V2

RS_R3_1 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('tanh', 1), ('tanh', 3), ('sigmoid', 0), ('sigmoid', 2), ('relu', 0), ('identity', 2)], concat=range(1, 9))
RS_R3_2 = Genotype(recurrent=[('tanh', 0), ('identity', 1), ('relu', 0), ('sigmoid', 1), ('sigmoid', 0), ('tanh', 2), ('identity', 2), ('identity', 3)], concat=range(1, 9))
RS_R3_3 = Genotype(recurrent=[('relu', 0), ('identity', 1), ('tanh', 0), ('tanh', 0), ('relu', 2), ('sigmoid', 0), ('identity', 4), ('identity', 2)], concat=range(1, 9))
RS_R3_4 = Genotype(recurrent=[('identity', 0), ('relu', 1), ('tanh', 0), ('relu', 0), ('sigmoid', 1), ('identity', 1), ('sigmoid', 2), ('identity', 4)], concat=range(1, 9))
