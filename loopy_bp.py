import os
import numpy as np
import pandas as pd
from graph_utils import *
import copy
import time
from scipy import special
import csv

########################################### get rotamer coordinates #######################################
directory = "top500-angles/rotasamp/"
rota_coords = []
radii = {"C":1.6, "O":1.3, "N":1.3, "S":1.7}
T=1

filenames = [f for f in os.listdir(directory) if f.endswith('.list')]
aas = [f.split('.')[0] for f in filenames]
data = {aas[a]: pd.read_csv(os.path.join(directory, filenames[a]), sep=':', comment='#', header=None) for a
                     in range(len(aas))}
max_indices = {k: v.nlargest(25, len(v.columns) - 2).index for (k, v) in data.items()}

def not_H(annot): 
    if "H" in annot: 
        if "OH" in annot or "NH" in annot or "CH" in annot: 
            return True
        return False
    return True

rot_dist_tables = {}
# first load in each file and keep only non-H atoms with annotated atom type at the end. 
for filename in os.listdir(directory):
    if filename.endswith(".pdb"):
        f = directory+filename
        aa = filename[:3]

        dist_table = []
        with open(f, 'r') as f:
            for line in f:                 
                list_elems = line.strip().split(' ')
                # print(list_elems)
                line = [elem for elem in list_elems if elem != ""]
                # print(line)

                # taking only top 100 rotamers and only non-H atoms
                if (int(line[4]) not in max_indices[aa]) or not(not_H(line[2])):
                    continue

                # clean data to handle missing spaces in input file
                if len(line) != 10: 
                    z_vec = line[-2].split(".")
                    decs = z_vec[1][:3]
                    new_z_vec= z_vec[0] + "." + decs
                    line[-2] = new_z_vec
                # print(line)

                line = line[:8] # keep only 3D coords

                atom = line[2][0]
                line.append(atom) # write atom type for easier lookup of radii

                # write backbone/sidechain atom type
                if line[2] == "N" or line[2] == "CA" or line[2] == "C" or line[2] == "O": 
                    line[0] = "bb"
                else: 
                    line[0] = "sc"

                # add to the table for this aa type
                if line[2] == "N": 
                    dist_table.append([])
                dist_table[-1].append(line)

        rot_dist_tables[aa] = dist_table

def atomic_energy(atom1, atom2):
    dist_1 = np.asarray(atom1[5:8], dtype=np.float64)
    dist_2 = np.asarray(atom2[5:8], dtype=np.float64)
    radius = radii[atom1[-1]] + radii[atom2[-1]] 

    dist = np.linalg.norm(dist_1-dist_2)
    if dist > radius:
        return 0
    elif dist >= 0.8254 * radius:
        return (-57.2738) * (dist / radius) + 57.2738
    return 10
 
####################################### get unary potential ##################################################
# compute unary potentials for each rotamer of each aa
def unary_pots(aa):
    table_unary_pots = np.zeros((len(aa),1))
    for r in range(len(aa)):
        bb_list = []
        sc_list = []

        for atom in aa[r]: 
            if atom[0] == "bb": 
                bb_list.append(atom)
            elif atom[0] == "sc":
                sc_list.append(atom)

        for atom1 in bb_list: 
            for atom2 in sc_list:
                energy = atomic_energy(atom1, atom2)
                table_unary_pots[r] += energy
    return -table_unary_pots/T

unary_pots_dict = {}
for key, val in rot_dist_tables.items():
    unary_pots_dict[key] = unary_pots(val)
    # print(key)

# print(rot_dist_tables["arg"])

########################################### create graph and add edge potentials ######################
# sc_dist_tables = copy.deepcopy(rot_dist_tables)

sc_dist_tables = {}
# # keep only sidechain coordinates
for key, res in rot_dist_tables.items():
    sc_dist_tables[key] = []
    for rot in res: 
        sc_list = []
        for atom in rot: 
            if atom[0] =="sc":
                sc_list.append(atom)
        sc_dist_tables[key].append(sc_list)


# print(len(sc_dist_tables["val"]))

def get_edge_pots(aa1, aa2):
    table_edge_pots = np.zeros((len(aa1), len(aa2)))
    for r1, rot1 in enumerate(aa1):
        for r2, rot2 in enumerate(aa2):
            for atom1 in rot1:
                for atom2 in rot2:
                    energy = atomic_energy(atom1, atom2)
                    table_edge_pots[r1][r2] += energy

    return table_edge_pots

def transform(coords, rots):
    new_rots = np.asarray(rots, dtype=str)
    new_rots_a = new_rots[:,:,5:8].astype(np.float) + coords

    new_rots_c = np.asarray(rots, dtype=str)
    new_rots_c[:,:,5:8] = new_rots_a

    return new_rots_c



def get_graph(pID):

    protein = "graph_gen/"+pID
    coords = read_coordinates_file(protein + ".txt")
    aas, edges = read_distances_file(protein + ".csv")
    old_nodes = create_graph_for_example(aas, coords, edges)

    # delete nodes which don't have variable rotamer states
    nodes=[]
    idx_to_add=[]
    for i,node in enumerate(old_nodes):
        idx_to_remove=[]
        for j,child in enumerate(node.children):
            if child.aa.lower() not in sc_dist_tables.keys():
                idx_to_remove.append(j)
        for j in sorted(idx_to_remove, reverse=True):
            del node.children[j]
        if node.aa.lower() not in sc_dist_tables.keys():
            idx_to_remove.append(i)
        else:
            nodes.append(node)

    # loop through children of each node
    for node in nodes: 
        node_aa = node.aa
        if node.rots is None:
            node_rots = sc_dist_tables[node_aa.lower()]
            node.rots = transform(node.coord, node_rots)
        for c, child in enumerate(node.children):
            child_aa = child.aa
            # lookup up the rotamer table for these aa's 
            if child.rots is None:
                child_rots = sc_dist_tables[child_aa.lower()]
                child.rots = transform(child.coord, child_rots)            
            node.edge_pots.append(get_edge_pots(node.rots, child.rots))

        node.edge_pots = -np.array(node.edge_pots)/T

    return nodes

criterion = 1e-10

def get_beliefs(nodes):
    beliefs = np.zeros((len(nodes), 25))
    for n, node in enumerate(nodes): 
        beliefs[n] = np.sum(node.incoming_messages, axis=0)
    return beliefs/np.sum(beliefs)

def compute_mean_change(arrs1, arrs2):
    change = np.sum(np.abs(arrs1-arrs2), axis=1)
    return np.mean(change)

####################################### belief propagation ###############################################
def loopy_bp_one_iter(nodes):
    for node in nodes:
        for c, child in enumerate(node.children):
            # send message
            outgoing_message = np.sum(node.incoming_messages, axis=0) + node.edge_pots[c]
            # outgoing message with all included
            outgoing_msg_wo_rem = special.logsumexp(outgoing_message, axis=0).T
            # remove the message sent by this child
            o_sub = node.incoming_messages[child.idx]
            child.incoming_messages[node.idx] = outgoing_msg_wo_rem-o_sub

            #receive message
            incoming_message = np.sum(child.incoming_messages, axis = 0) + node.edge_pots[c].T
            # incoming message with all included
            incoming_message_wo_rem = special.logsumexp(incoming_message, axis = 0).T
            # remove the message sent by the parent
            i_sub = child.incoming_messages[node.idx]
            node.incoming_messages[child.idx] = incoming_message_wo_rem - i_sub
             
def loopy_bp(n_itrs, nodes):
    # initialize messages and nary potentials (message to self)
    deltas = []
    for n, node in enumerate(nodes): 
        node.idx = n
        node.incoming_messages = np.zeros((len(nodes), 25))
        node.incoming_messages[node.idx] = unary_pots_dict[node.aa].squeeze(-1)
    
    old_beliefs = get_beliefs(nodes)
    for i in range(n_itrs):
        loopy_bp_one_iter(nodes)
        new_beliefs = get_beliefs(nodes)
        delta = compute_mean_change(new_beliefs, old_beliefs)
        deltas.append(delta)
        old_beliefs = new_beliefs

    return deltas

def loopy_bp_inference(n_itrs,nodes):
    deltas = loopy_bp(n_itrs, nodes)  
    prediction = []
    for node in nodes:
        belief = np.sum(node.incoming_messages, axis = 0)
        belief = belief/np.sum(belief)
        max_rots = np.argwhere(belief==np.max(belief)).flatten().tolist()
        list_preds = [rot_dist_tables[node.aa][pred][0][4] for pred in max_rots]
        prediction.append(list_preds)
    return prediction, deltas

def predict(bg_idx=0,end_idx=7000):
    with open('./preds_100_8/log.txt', 'w') as log:
        n_itrs=50
        with open('fileIDs.txt', 'r') as f:
            IDs=f.readline().split(',')
            IDs=IDs[bg_idx:end_idx]

        for pt_id in IDs:
            pt_id=pt_id.strip()
            print(f'processing {pt_id}', end='\r')
            if (pt_id == "1AGT"):
                try:
                    nodes=get_graph(pt_id)
                    prediction, deltas = loopy_bp_inference(n_itrs,nodes)
                    delimiter='\t'
                    with open(f'./preds_100_8/{pt_id}', 'w') as f:
                        f.write(delimiter.join([" ".join(p) for p in prediction]))
                    with open("deltas.txt","a+") as d:
                        d.writelines("%s\t" % delta for delta in deltas)
                        d.write("\n")
                except OSError as e:
                    log.write(pt_id+'\n')
        print()

if __name__ == "__main__":
    bg_idx,end_idx=0,100
    try:
        os.remove('deltas.txt')
    except:
        pass
    predict()
