import numpy as np


def loaddata(filename):
    with open(filename, "r") as infile:
        structlist = []
        newstruct = []
        fusedstruct = []
        labellist = []
        for line in infile:
            col = line.split()
            if len(col) > 6:
                newatom = [float(x) for x in col[1:]]
                newstruct.append(newatom)
                fusedstruct.append(newatom)
            elif len(col) == 4:
                label = float(col[1])
                if len(newstruct) > 0:
                    labellist.append(label)
                    structlist.append(np.array(newstruct, dtype=np.float64))
                    newstruct = []
            else:
                continue
        fusedstruct = np.array(fusedstruct, dtype=np.float64)
        labellist = np.array(labellist, dtype=np.float64)
    return structlist, fusedstruct, labellist


if __name__ == "__main__":
    structlist, fusedstruct, labellist = loaddata("function.data")




