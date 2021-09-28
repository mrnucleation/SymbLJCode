import sys
from NNetCalc import runsimtest
try: filename = sys.argv[1]
except IndexError: filename = "dumpfile.dat"



with open(filename, "r") as infile, open("score.test", "w") as outfile, open("score.train", "w") as outfile2:
    minval = 1e300
    minval_test = 1e300
    coords = ""
    coords_test = ""
    cnt = 0
    minloc = 0
    minloc_test = 0
    for i, line in enumerate(infile):
        cnt += 1
        col = line.split("|")
        eng = float(col[-1])
        try:
            eng_test = float(col[-2])
        except:
            eng_test = 0.0
        coords = [float(x) for x in col[0].split()]
        if minval > eng:
            minval = eng
            minloc = i+1
        eng_test = runsimtest(coords)
        if minval_test > eng_test:
            minval_test = eng_test
            minloc_test = i+1
            coords_test = [float(x) for x in col[0].split()]
        outfile.write("%s %s\n"%(i+1, minval_test))
        outfile2.write("%s %s\n"%(i+1, minval))

print("Lowest Train Value: %s"%(minval))
print("Lowest Test Value: %s"%(minval_test))
with open("Parameters_Test.dat", "w") as outfile:
    for x in coords_test:
        outfile.write("%s\n"%(x))
print("Number of Total Evaluations: %s"%(cnt))
print("Number of Evaluations till Train Minima was found: %s"%(minloc))
print("Number of Evaluations till Test Minima was found: %s"%(minloc_test))
