import torch
import torch.nn as nn
from copy import deepcopy
# from ..cell_operations import ResNetBasicblock
# from .search_cells     import NAS201SearchCell as SearchCell
# from .genotypes        import Structure
from cell_operations import OPS,ResNetBasicblock,NAS_BENCH_201

class NAS201SearchCell(nn.Module):

    def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=False):
        super(NAS201SearchCell, self).__init__()

        self.op_names  = deepcopy(op_names)
        self.edges     = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim    = C_in
        self.out_dim   = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    xlists = [OPS[op_name](C_in , C_out, stride, affine, track_running_stats) for op_name in op_names]
                else:
                    xlists = [OPS[op_name](C_in , C_out,      1, affine, track_running_stats) for op_name in op_names]
                self.edges[ node_str ] = nn.ModuleList( xlists )
        self.edge_keys  = sorted(list(self.edges.keys()))
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
        self.num_edges  = len(self.edges)
        
    def forward(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure[i-1]
            inter_nodes = []
            for j,op_id in enumerate(cur_op_node):
                
                node_str = '{:}<-{:}'.format(i, j)
                inter_nodes.append( self.edges[node_str][op_id]( nodes[j] ) )
            nodes.append( sum(inter_nodes) )
        return nodes[-1]

class TinyNetwork(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
        super(TinyNetwork, self).__init__()
        self._C        = C
        self._layerN   = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C,affine = affine,track_running_stats=track_running_stats))

        layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = NAS201SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
                else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append( cell )
            C_prev = cell.out_dim
        
        self.op_names   = deepcopy( search_space )
        self._Layer     = len(self.cells)
        self.edge2index = edge2index
        self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev,affine = affine,track_running_stats=track_running_stats), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_weights(self):
        xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
        xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
        xlist+= list( self.classifier.parameters() )
        return xlist
    


    def forward(self, inputs, structure):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, NAS201SearchCell):
                feature = cell(feature, structure)
            else:
                feature = cell(feature)

        

        out = self.lastact(feature)
        out = self.global_pooling( out )
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits