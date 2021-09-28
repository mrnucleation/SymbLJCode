import PyQt5

from ete3 import Tree, TreeStyle, NodeStyle
from math import fabs
import os

infile = open("treestructure.nh", "r")
anafile = open("analytics.dat", "r")
minenergy = -8.0


workDir = os.getcwd()
imageDir = workDir + "/images/"
if not os.path.exists(imageDir):
    os.mkdir(imageDir)

maxnodesize = 100.0
for i, line in enumerate(infile):
#    t = Tree("treestructure.nh", format=1)
#    t = Tree("treestructure.nh", format=1)
    treestruct = line.split()[0]
    print treestruct
    t = Tree(line)

    nodescores = {}
    while True:
        aline = anafile.readline()
        col = aline.split()
        if len(col) < 1 or aline is None:
            break
        nodeid = int(col[0])
#        score = float(col[1])
#        wins = int(col[2])
        energy = float(col[1])
#        print nodeid, score, wins, energy
        nodescores[nodeid] = energy/5.0

    if i%5 != 0:
        continue



    minenergy = -1e30
    for key in nodescores:
        energy = nodescores[key]
        if energy < minenergy:
            minenergy = energy


    slope = (maxnodesize-10.0)/(0.0-5.0)
    intercept = 100.0


    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.rotation = 90
    ts.scale = 120
    ts.show_branch_length = True
    ts.show_branch_support = True

    for node in t.traverse():
        nstyle = NodeStyle()
        nstyle["shape"] = "sphere"
        try:
            nodeid = int(node.name)
            energy = nodescores[nodeid]

#            ratio = 1000*(minenergy-energy) + 50*energy
#            nstyle["size"] = 100*ratio
            size = energy*slope+intercept
            if size < 10:
                size = 10
            nstyle["size"] = size

            print(size)
        except ValueError:
            nstyle["size"] = 100
        node.set_style(nstyle)
#    t.show(tree_style=ts)
#    t.render(imageDir+"tree%s.png"%(i), units="px", tree_style=ts)
t.show(tree_style=ts)
