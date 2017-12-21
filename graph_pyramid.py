import argparse
import numpy as np
from PIL import Image

from grid_graph import GridGraph
from disjoint_set import DisjointSet

class GraphPyramid():
    def __init__(self, h, w, edge_weights):
        self.shape = (h, w)
        self.weights = edge_weights
        self.grid_graph = GridGraph(h, w, edge_weights)
        self.segment = DisjointSet(h * w)
        self.pyramid = self._build_pyramid()

    def _build_pyramid(self):
        # Build a hierarchy of partitions from the base graph.
        for k in range(20): # TODO: Check termination condition.
            contraction_kernel = set()
            # Find minimum weighted edge from each vertex and
            # add qualified edges to contraction edge set.
            min_edges = set()
            for v in self.grid_graph.vertices():
                neighbors = {d.tail : d.weight
                             for d in self.grid_graph.primal[v].values()}
                if len(neighbors) == 0:
                    continue
                # Add all minimum edges in.
                min_edge_val = min(neighbors.values())
                for u in neighbors.keys():
                    if neighbors[u] == min_edge_val:
                        if (min(u, v), max(u, v)) in min_edges:
                            # TODO: Test Ext(CC) < P*Int(cc)
                            contraction_kernel.add((min(u, v), max(u, v)))
                        else:
                            min_edges.add((min(u, v), max(u, v)))
            if len(contraction_kernel) == 0:
                break
            print "iteration %d: #edge contracted=%d" % \
                    (k, len(contraction_kernel))
            # Contract edges in contraction kernel.
            for (u, v) in contraction_kernel:
                if self.segment.root(u) != self.segment.root(v):
                    #print 'contracting:', u, v
                    self.grid_graph.contract_edge(
                            self.segment.root(u), self.segment.root(v))
                    self.segment.unite(u, v)
                assert(len(set(self.segment.array())) == len(self.grid_graph.primal))
            # TODO: Update edge attributes to be the smallest edge merged in.
            print np.array(self.segment.array()).reshape(self.shape)
        return self.segment.array()

def convert_image_to_weighted_edges(image):
    weighted_edges = {v : {} for v in range(image.size)}
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            v = i * w + j
            if j < w - 1:
                weighted_edges[v][v + 1] = abs(image[i][j + 1] - image[i][j])
            if i > 0:
                weighted_edges[v][v - w] = abs(image[i - 1][j] - image[i][j])
            if j > 0:
                weighted_edges[v][v - 1] = abs(image[i][j - 1] - image[i][j])
            if i < h - 1:
                weighted_edges[v][v + w] = abs(image[i + 1][j] - image[i][j])
    return weighted_edges

def plot_segment(image, segment, path):
    segment = np.array(segment).reshape(image.shape)
    gx, gy = np.gradient(segment)
    g = np.logical_or(gx.astype(np.bool), gy.astype(np.bool))
    result = Image.fromarray(g.astype(np.uint8) * 255)
    result.save('dgc_' + path)

def main(args):
    image = Image.open(args.image_path).convert('L')
    image = np.array(image, dtype=np.float32) / np.max(image)
    print image.shape
    #image = np.eye(5).repeat(2, axis=1).repeat(2, axis=0)
    #image = np.r_[image, image]
    weighted_edges = convert_image_to_weighted_edges(image)
    #print weighted_edges
    pyramid = GraphPyramid(image.shape[0], image.shape[1], weighted_edges)
    segment = pyramid.pyramid
    #print np.array(segment).reshape(image.shape)
    plot_segment(image, segment, args.image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    args, unparsed = parser.parse_known_args()
    main(args)
