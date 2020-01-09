import os
import numpy as np
import matplotlib.pyplot as plt

pdb_id = "1AGT"

pdb_data_file = "pdb-data/" + pdb_id + " _angles.txt"
rotamer_pred_file = "preds_100_8/" + pdb_id
rots_dir = "rotasamp/"

# extract sequence of rotamer predictions
with open(rotamer_pred_file, 'r') as r:
    rots = r.readline().strip().split('\t')
    for idx, rot in enumerate(rots):
        rot = rot.split(' ')
        rots[idx] = int(rot[0])
        
    
# extract pdb data as numpy
pdb_data = np.loadtxt(pdb_data_file, dtype=str, skiprows=1)
del_rows = np.where(pdb_data[:,3]=="NA")[0]

pdb_data = np.delete(pdb_data, del_rows, axis=0)
aa_seqs = pdb_data[:,0]

aa_counts = {}
pdb_chi_angles = pdb_data[:, 3:]
nas = np.argwhere(pdb_chi_angles=="NA")
for idx, row in enumerate(pdb_chi_angles):
    stop = np.argwhere(row=="NA")[0]
    aa_counts[aa_seqs[idx][:3].lower()] = stop.item()

pdb_chi_angles = pdb_data[:, 3:].flatten()
nas = np.argwhere(pdb_chi_angles=="NA")
pdb_chi_angles = np.delete(pdb_chi_angles, nas).astype(np.float)

# extract predicted chi angles 
pred_chi_angles = []
for idx, rot in enumerate(rots):
    aa = aa_seqs[idx][:3].lower()
    fname = rots_dir + aa + ".list"
    aa_data = np.loadtxt(fname, delimiter=":")
    if aa_data[0][0] == "#":
        aa_data = aa_data[1:]
    aa_rot = aa_data[rot,:aa_counts[aa]]
    pred_chi_angles.append(aa_rot)
pred_chi_angles=[val for sublist in pred_chi_angles for val in sublist]


plt.plot(pred_chi_angles,'bo', label="Predicted angles")
plt.plot(pdb_chi_angles, 'ro', label="Actual angles")
plt.legend()
plt.title("Comparison of predicted vs actual conformations")
plt.xlabel("Sequence position")
plt.ylabel("Chi angle")
plt.show()