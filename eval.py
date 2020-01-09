import numpy as np
import glob
import os
import re

DIR_PRED = "prediction/"
DIR_TARGET = "target/"


def eval(dir_pred=DIR_PRED, dir_target=DIR_TARGET):
    fnames_pred = []
    os.chdir(dir_pred)
    for fname in glob.glob("*"):
        if re.match("^[\w\d]*$", fname):
            fnames_pred.append(fname)
    os.chdir("../")
    acc_list = []
    with open("acc.txt", "w") as g:
        for fname in fnames_pred:
            with open(dir_pred + fname, "r") as f:
                pred = f.readline().strip().split(",")
            with open(dir_target + fname, "r") as f:
                target = f.readline().strip().split(",")
            acc = eval_two_list(pred, target)
            acc_list.append(acc)
            g.write(f"{fname} {acc}\n")
        g.write(f"avg_acc {sum(acc_list) / len(acc_list)}\n")


def eval_two_list(list1, list2):
    acc = 0.0
    if len(list1) != len(list2):
        raise Exception("wrong length")
    for i, e1 in enumerate(list1):
        e2 = list2[i]
        if e1 == e2:
            acc += 1
    return acc / len(list1)


if __name__ == "__main__":
    eval()
