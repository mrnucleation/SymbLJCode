import sys

try: filename = sys.argv[1]
except IndexError: filename = "dumpfile.dat"



with open(filename, "r") as infile, open("score.test", "w") as outfile, open("score.train", "w") as outfile2, open("score.total", "w") as outfile3:
    minval_train = 1e300
    minval_test = 1e300
    minval_total = 1e300
    coords_total = ""
    coords_train = ""
    coords_test = ""
    cnt = 0
    minloc = 0
    minloc_test = 0
    for i, line in enumerate(infile):
        cnt += 1
        col = line.split("|")
        eng_total = float(col[-1])
        eng_train = float(col[-2])
        eng_test = float(col[-3])
        if minval_train > eng_train:
            minval_train = eng_train
            minloc_train = i+1
            coords_train = col[0]
        if minval_test > eng_test:
            minval_test = eng_test
            minloc_test = i+1
            coords_test = col[0]
        if minval_total > eng_total:
            minval_total = eng_total
            minloc_total = i+1
            coords_total = col[0]
        outfile.write("%s %s\n"%(i+1, minval_test))
        outfile2.write("%s %s\n"%(i+1, minval_train))
        outfile3.write("%s %s\n"%(i+1, minval_total))

print("Lowest Train Value: %s"%(minval_train))
print("Lowest Test Value: %s"%(minval_test))
print("Lowest Total Value: %s"%(minval_total))
if len([float(x) for x in coords_train.split()]) < 15:
    print("Lowest Coords: %s"%(coords))
with open("Parameters.dat", "w") as outfile:
    for x in [float(x) for x in coords_train.split()]:
        outfile.write("%s\n"%(x))
with open("Parameters_Test.dat", "w") as outfile:
    for x in [float(x) for x in coords_test.split()]:
        outfile.write("%s\n"%(x))
with open("Parameters_Total.dat", "w") as outfile:
    for x in [float(x) for x in coords_total.split()]:
        outfile.write("%s\n"%(x))
print("Number of Total Evaluations: %s"%(cnt))
print("Number of Evaluations till Train Minima was found: %s"%(minloc_train))
print("Number of Evaluations till Test Minima was found: %s"%(minloc_test))
print("Number of Evaluations till Total Minima was found: %s"%(minloc_total))
