import json


def read_distances_file(filename, dist_threshold=8):
    edges = []
    aas = []
    with open(filename) as f:
        lines = f.readlines()
    for j, line in enumerate(lines):
        aa, distances = line.split(' ', 1)
        aas.append(aa)
        distances = json.loads(distances)
        edge_ids = []
        for idx, dist in enumerate(distances):
            if dist < dist_threshold:
                edge_ids.append(j + idx + 1)
        edges.append(edge_ids)
            
    return aas, edges

def parse_array(text):
    text = text.strip()[1: -1].split()
    return [float(x) for x in text]

def read_coordinates_file(filename):
    locs = []
    aas = []
    with open(filename) as f:
        lines = f.readlines()
    for j, line in enumerate(lines):
        aa, loc = line.split(' ', 1)
        aas.append(aa)
        locs.append(parse_array(loc))     
    return locs

class Node:
    def __init__(self, aa, coord):
        self.aa = aa.lower()
        self.children = [] # outgoing edges only
        self.edges = None # defined only for root
        self.coord = coord
        self.edge_pots = []
        self.rots = None
        self.incoming_messages = None
        self.idx = None

def create_graph_for_example(aas, coords, edges):
#     edge_dict = {}
    nodes = [Node(aa, coord) for (aa, coord) in zip(aas, coords)]
    for n, edge_ids in enumerate(edges):
        for e in edge_ids:
            nodes[n].children.append(nodes[e])
    nodes[0].edges = edges
    return nodes # root