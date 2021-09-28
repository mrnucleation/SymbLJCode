import sys
from NNetCalc import runsim_test

try: filename = sys.argv[1]
except IndexError: filename = "dumpfile.dat"



with open(filename, "r") as infile, open("score.test", "w") as outfile, open("dumpfile_cross.dat", "w") as outfile2:
    mintest = 1e300
    mincoords = []
    minloc = 0
    cnt = 0

    for i, line in enumerate(infile):
        cnt += 1
        col = line.split("|")
        coords_str = col[0]
        coords = [float(x) for x in coords_str.split()]
        eng = float(col[-1])

        eng_test = runsim_test(coords)
        score = runsim(parameters, verbose=True, usemask=False)
        if mintest > eng_test:
            mintest = eng_test
            mincoords = coords
            minloc = i+1
        outfile.write("%s %s\n"%(i+1, mintest))
        outfile2.write("%s | %s | %s \n"%(coords_str, eng_test, eng))

print("Lowest Test Value: %s"%(mintest))
with open("Parameters_Test.dat", "w") as outfile:
    for x in mincoords:
        outfile.write("%s\n"%(x))
print("Number of Total Evaluations: %s"%(cnt))
print("Number of Evaluations till Test Minima was found: %s"%(minloc))
