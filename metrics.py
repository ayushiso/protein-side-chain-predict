import os
import numpy as np
import matplotlib.pyplot as plt

# preds will be list for each node while targs will be a number for each node. 

# compute acc for one protein
def one_prot(preds, targs):
    acc = 0.0
    with open(preds, 'r') as p:
        preds_arr = p.readline().split('\t')
        for idx, pred in enumerate(preds_arr):
            preds_arr[idx] = pred.split(' ')
    with open(targs, 'r') as t:
        targs_arr = t.readline().split(',')
    non_fixed_targs = []
    for targ in targs_arr:
        if targ != "-2":
            non_fixed_targs.append(targ)

    if len(non_fixed_targs) != len(preds_arr):
        print(len(targs_arr))
        print(len(preds_arr))
        raise Exception("Length mismatch between prediction and ground truth")

    count = 0.0
    for t, targ in enumerate(non_fixed_targs):
        if targ in preds_arr[t]:
            acc += 1.0
        if targs != -1:
            count += 1.0

    print(acc/count)
    return acc/count

if __name__ == "__main__":
    preds_dir = "preds_100/"
    targs_dir = "nothing_but_the_truth/"

    with open("fileIDs.txt", 'r') as f:
        fileIDs = f.readline().split(", ")
        fileIDs = fileIDs[:100]
    total_acc = 0.0
    for pID in fileIDs:
        pred_f = preds_dir + pID
        targ_f = targs_dir + pID +".txt"

        total_acc += one_prot(pred_f, targ_f)

    print("total acc=", total_acc/100)

    # generate plots for convergence of loopy bp (deltas)

    # deltas = np.loadtxt("deltas.txt")
    # for line in deltas:
    #     plt.plot(line)
    # plt.xlabel("Iterations of loopy BP")
    # plt.ylabel("Mean change of beliefs in one iteration")
    # plt.title("Change in beliefs with iterations")

    # plt.show()