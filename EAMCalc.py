import os
import sys
import numpy as np
from lammps import lammps
import ase

cmds = ["-screen", "log.screen"]
lmp = lammps(cmdargs=cmds)
#========================================================
def eamenergies(geofile=None, struct=None, outdatafile=None):
    lmp.command("clear")
    lmp.command("log log.equil")
    lmp.command("dimension 3")
    lmp.command("box tilt large")
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map array")
    lmp.command("boundary p p p ")
    if geofile is not None:
        lmp.command("read_data %s"%(geofile))
    else:
        xmin = struct[:,0].min() - 50.0
        xmax = struct[:,0].max() + 50.0
        ymin = struct[:,1].min() - 50.0
        ymax = struct[:,1].max() + 50.0
        zmin = struct[:,2].min() - 50.0
        zmax = struct[:,2].max() + 50.0
        lmp.command("region mybox block %s %s %s %s %s %s"%(xmin, xmax, ymin, ymax, zmin, zmax))
        lmp.command("create_box 1 mybox")
        lmp.command("mass 1 1.0")
        for iAtom in struct:
            x,y,z = tuple(iAtom)
            lmp.command("create_atoms 1 single %s %s %s"%(x,y,z))
    if outdatafile is not None:
        lmp.command("write_data %s"%(outdatafile))

    lmp.command("pair_style eam")
    lmp.command("neighbor 12.0 bin")

    lmp.command("pair_coeff * * Al.eam")
#    lmp.command("pair_coeff * * ../Al.eam")
            
    lmp.command('variable potential equal pe/atoms')
    lmp.command('variable natoms equal atoms')
    lmp.command("run 0 pre no")

    natoms = int(lmp.extract_variable('natoms', None, 0))
    value = lmp.extract_variable('potential', None, 0)

    return value
#========================================================
def writedump(struct, outdatafile):
    lmp.command("clear")
    lmp.command("log log.equil")
    lmp.command("dimension 3")
    lmp.command("box tilt large")
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map array")
    lmp.command("boundary p p p ")
    xmin = struct[:,0].min() - 50.0
    xmax = struct[:,0].max() + 50.0
    ymin = struct[:,1].min() - 50.0
    ymax = struct[:,1].max() + 50.0
    zmin = struct[:,2].min() - 50.0
    zmax = struct[:,2].max() + 50.0
    lmp.command("region mybox block %s %s %s %s %s %s"%(xmin, xmax, ymin, ymax, zmin, zmax))
    lmp.command("create_box 1 mybox")
    lmp.command("mass 1 1.0")
    for iAtom in struct:
        x,y,z = tuple(iAtom)
        lmp.command("create_atoms 1 single %s %s %s"%(x,y,z))
    lmp.command("write_data %s"%(outdatafile))
#========================================================
def readdump(indatafile):
    atoms = ase.io.read(indatafile, format='lammps-data',style='atomic', units='metal')
    positions = atoms.get_positions()
    return positions



