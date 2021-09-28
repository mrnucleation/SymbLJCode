import os
import subprocess

import numpy as np
from ase.io import read
import re

idKey2 = re.compile(r'\s+[-+]?\d*\.\d')

def shell(command):
    return subprocess.check_output(command,shell=True).strip()

def getPotcar(outcar):
    for line in open(outcar):
        rec = line.strip()
        if rec.startswith('TITEL'):
            potcar = line.split('=')[-1].strip()
    return potcar        

def getEdiff(outcar):
    diff = []
    for line in open(outcar):
        rec = line.strip()
        if rec.startswith('EDIFF'):
            ediff = line.split('=')[-1].strip()
            diff.append(ediff)
    return diff[0].replace('stopping-criterion for ELM', '').strip()        


def checkFile(path):
    if os.path.exists(path) and os.path.getsize(path) > 0: 
        flag = True
    else:
        flag = False   
    return flag

def getNELM(OSZICAR):
    nElm = shell("tail -2  " + OSZICAR + "| head -1 |awk '{print $2}'").decode("utf-8")
    return nElm

def checkSuccess(path):
    search = 'General timing and accounting informations for this job:\n'
    with open(path) as f:
        datafile = f.readlines()
    success = False
    for line in datafile:
        if search in line:
            success = True
            break
    return success

def getTotalEnergy(OUTCAR):
    energy = shell("grep 'free  energy' " +  OUTCAR +  "|awk '{print $5}'|tail -n 1").decode("utf-8")
    print(energy)
    try:
        outvar = float(energy)
        return float(energy)
    except:
        return energy


def getPositions(nAtoms, OUTCAR):
    file = open(OUTCAR, 'r')
    lines= file.readlines()
    file.close()
    n=0

def getPosForces(OUTCAR):
    positions = []
    forces = []
    with open(OUTCAR, "r") as infile:
        lines = infile.readlines()
        n = 0
        pos = -1
        for i, line in enumerate(lines):
            if 'TOTAL-FORCE' in line:
                pos = i
        infile.seek(0)

        for i, line in enumerate(lines):
            if i < pos+2:
                continue
            rawid = idKey2.search(line)
            if rawid is not None:
                x,y,z, fx,fy,fz = tuple([float(x) for x in line.split()])
                positions.append([x,y,z])
                forces.append([fx,fy,fz])
                pass
            else:
                break
    return np.array(positions), np.array(forces)

def getPositions(OUTCAR):
    pos, forces = getPosForces(OUTCAR)
    return pos


def getForces(OUTCAR):
    pos, forces = getPosForces(OUTCAR)
    return forces

def totalAtoms(poscar):
    with open(poscar, 'r') as poscar:
        poscar_lines = poscar.readlines()
    stoichiometries = poscar_lines[6].split()
    n_atoms = 0
    for stoichiometry in stoichiometries:
        n_atoms += int(stoichiometry)
    return n_atoms    

def writePotcar(poscar):
    potcarFile = poscar.replace('POSCAR', 'POTCAR')
    struct = read(poscar, -1, 'vasp')
    Species = []
    for x in struct.get_chemical_symbols():
        if x not in Species:
            Species.append(x)

    potcars = []
    potcarSrc = '/home/share/cnm50256/PotcarDirs/'
    
    for sp in Species:
        potcars.append(potcarSrc + sp + '/POTCAR')
    
    with open(potcarFile, 'w') as outfile:
        for potcar in potcars:
            with open(potcar) as infile:
                for line in infile:
                    outfile.write(line)

    print ('Generating potcar with ', Species, 'at ', potcarFile)


def elastic_moduli(path):
    from re import M as multline
    from re import findall
    from numpy import array
    regex = r"\s*TOTAL\s+ELASTIC\s+MODULI\s+\(kBar\)\s*\n"                     \
            r"\s*Direction\s+XX\s*YY\s*ZZ\s*XY\s*YZ\s*ZX\s*\n"                 \
            r"\s*-+\s*\n"                                                      \
            r"\s*XX\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\n"      \
            r"\s*YY\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\n"      \
            r"\s*ZZ\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\n"      \
            r"\s*XY\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\n"      \
            r"\s*YZ\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\n"      \
            r"\s*ZX\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\n"      \
            r"\s*-+\s*\n"
    result = findall(regex,open(path).read(), multline)[0]
    return array(result,dtype='float64').reshape((6,6))

def corrCij(Cij):
    C_ij = np.zeros((6,6))
    C_ij[0,0] = Cij[0,0]*0.1
    C_ij[0,1] = Cij[0,1]*0.1
    C_ij[0,2] = Cij[0,2]*0.1
    C_ij[1,0] = Cij[1,0]*0.1
    C_ij[1,1] = Cij[1,1]*0.1
    C_ij[1,2] = Cij[1,2]*0.1
    C_ij[2,0] = Cij[2,0]*0.1
    C_ij[2,1] = Cij[2,1]*0.1
    C_ij[2,2] = Cij[2,2]*0.1
    C_ij[0,3] = Cij[0,4]*0.1
    C_ij[0,4] = Cij[0,5]*0.1
    C_ij[0,5] = Cij[0,3]*0.1
    C_ij[1,3] = Cij[1,4]*0.1
    C_ij[1,4] = Cij[1,5]*0.1
    C_ij[1,5] = Cij[1,3]*0.1
    C_ij[2,3] = Cij[2,4]*0.1
    C_ij[2,4] = Cij[2,5]*0.1
    C_ij[2,5] = Cij[2,3]*0.1
    C_ij[3,0] = Cij[4,0]*0.1
    C_ij[4,0] = Cij[5,0]*0.1
    C_ij[5,0] = Cij[3,0]*0.1
    C_ij[3,1] = Cij[4,1]*0.1
    C_ij[4,1] = Cij[5,1]*0.1
    C_ij[5,1] = Cij[3,1]*0.1
    C_ij[3,2] = Cij[4,2]*0.1
    C_ij[4,2] = Cij[5,2]*0.1
    C_ij[5,2] = Cij[3,2]*0.1
    C_ij[3,3] = Cij[4,4]*0.1
    C_ij[3,4] = Cij[4,5]*0.1
    C_ij[3,5] = Cij[4,3]*0.1
    C_ij[4,3] = Cij[5,4]*0.1
    C_ij[4,4] = Cij[5,5]*0.1
    C_ij[4,5] = Cij[5,3]*0.1
    C_ij[5,3] = Cij[3,4]*0.1
    C_ij[5,4] = Cij[3,5]*0.1
    C_ij[5,5] = Cij[3,3]*0.1
    
    return C_ij






if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
#    getPosForces(filename)
    positions, forces = getPosForces(filename)
    print(len(forces))
    for position, force in zip(positions, forces):
        print(position, force)

