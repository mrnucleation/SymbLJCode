from math import cos, sin, exp, fabs, pi, sqrt, tanh, floor, pow


#------------------------------------------
def hyperpara(parameters, fileprint=True):
    f = np.sum(parameters**2)/len(parameters)
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))
    return 

#------------------------------------------
#  -10 < xi < 10
#  f = 0  at x=(1,1,1,1)
def Colville(parameters, fileprint=True):
    x1 = parameters[0]
    x2 = parameters[1]
    x3 = parameters[1]
    x4 = parameters[1]
    term = 100*(x1**2-x2)**2 + (x1-1)**2
    term += (x3-1)**2 + 90*(x3**2-x4)**2
    term += 10.1*(x2-1)**2 + 10.1*(x4-1)**2
    term += 19.8*(x2-1)*(x4-1)
    f = term
    with open("dumpfile.dat", "a") as outfile:
        outstr = ' '.join([str(x) for x in parameters])
        outfile.write('%s | %s \n'%(outstr, f))
    return f


#------------------------------------------
#  -10 < xi < 10
#  f = 0  at xi=(2**(-(2**i - 2)/2**i))
def DixonPrice(parameters, fileprint=True):
#    x1 = parameters[0]
#    x2 = parameters[1]
#    x3 = parameters[1]
#    x4 = parameters[1]
    term = 0.0
    term += (parameters[0]-1.0)**2

    for i, x in enumerate(parameters):
        if i == 0:
            continue
        term += (i+1)*(2.0*parameters[i]**2 - parameters[i-1])**2
    f = term
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))
    return f

#------------------------------------------
#  -10 < xi < 10, i=1,2
#  f = -1  at x=(0,0)
def CrossInTray(parameters, fileprint=True):
    x1 = parameters[0]
    x2 = parameters[1]
    term = fabs(sin(x1)*sin(x2)*exp(fabs(100- sqrt(x1**2+x2**2)/pi)))+1
    term = term**0.1
    f = -0.0001*term
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))
    return f
#------------------------------------------
#  -10 < xi < 10, i=1,2
#  f = -1  at x=(0,0)
def CrossLegTable(parameters, fileprint=True):
    x1 = parameters[0]
    x2 = parameters[1]
    term = fabs(sin(x1)*sin(x2)*exp(fabs(100- sqrt(x1**2+x2**2)/pi)))+1
    term = term**0.1
    f = -1.0/term
    with open("dumpfile.dat", "a") as outfile:
        outstr = ' '.join([str(x) for x in parameters])
        outfile.write('%s | %s \n'%(outstr, f))
    return f
#------------------------------------------
#  -5.12 < xi < 5.12, i=1,2
#  f = -1  at x=(0,0)
def DropWave(parameters, fileprint=True):
    x1 = parameters[0]
    x2 = parameters[1]
    rsq = x1**2 + x2**2
    f = 1.0+cos(12*sqrt(rsq))
    f = f/(0.5*rsq+2.0)
    f = -f
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))
    return f

#------------------------------------------
#  -512 < xi < 512
#  f = -959.6407  at x=(512, 404.2319)
def Eggholder(parameters, fileprint=True):
    x1 = parameters[0]
    x2 = parameters[1]
    f = -(x2+47.0)*sin(sqrt(fabs(x2+0.5*x1+47.0)))-x1*sin(sqrt(fabs(x1-(x2+47.0))))
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))
    return f


#------------------------------------------
# -10 < xi < 10
def HolderTable(parameters):
    x1 = parameters[0]
    x2 = parameters[1]
    f = -fabs(sin(x1)*cos(x2)*exp(fabs(1-sqrt(x1*x1 + x2*x2)/pi)))
    with open("dumpfile.dat", "a") as outfile:
        outstr = ' '.join([str(x) for x in parameters])
        outfile.write('%s | %s \n'%(outstr, f))
    return f
#------------------------------------------
#  -15 < x1 < -5,  -3 < x2 < 3
#  Global Minimal f=0  a x' = (-10, 1)
def Bukin(parameters, fileprint=True):
    x1 = parameters[0]
    x2 = parameters[1]
    f = 100.0*sqrt(fabs(x2-0.01*x1**2)) + 0.01*fabs(x1+10.0)
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))

#    print(x1,x2,f)
    return f
#------------------------------------------
#  0 < x1 < 14,  0 < x2 < 14
#  Global Minimal f=0  a x' = (2, 2)
def Damavandi(parameters, fileprint=True):
    x1 = parameters[0]
    x2 = parameters[1]
    f = 1.0 - fabs(sin(pi*(x1-2))*sin(pi*(x2-2))/(pi**2*(x1-2)*(x2-2)))**5
    f = f*(2+(x1-7)**2 + 2*(x2-7)**2)
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))
    return f
#------------------------------------------
#  1 < xi < 60
#  Global Minimal f=0  a x' = (53.81, 1.27, 3.012, 2.13, 0.507)
def DevilliersGlasser02(parameters, fileprint=True):
    x1 = parameters[0]
    x2 = parameters[1]
    x3 = parameters[2]
    x4 = parameters[3]
    x5 = parameters[4]
#    print(parameters)
    f = 0.0
    try:
        for i in range(1, 25):
            t_i = 0.1*(i-1)
            y_i = 53.81*(1.27**t_i)*tanh(3.012*t_i + sin(2.13*t_i))*cos(exp(0.507)*t_i)
            term = x1*x2**(t_i) * tanh(x3*t_i+sin(x4*t_i))*cos(t_i*exp(x5))-y_i
            term = term*term
            f += term
    except OverflowError:
        f = 1e300

    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))

#    print(x1,x2,f)
    return f

#------------------------------------------
def Rastrigin(parameters):   
    f = 10*len(parameters)
    for value in parameters:
        f += value**2 - 10*cos(2.0*pi*value)
    with open("dumpfile.dat", "a") as outfile:
        outstr = ' '.join([str(x) for x in parameters])
        outfile.write('%s | %s \n'%(outstr, f))
    return f 
#------------------------------------------

def DeJongQuartic(parameters, fileprint=True):
    """
    De Jong's quartic function:
    The modified fourth De Jong function, Equation (20) of [2]
    minimum is f(x)=random, but statistically at xi=0
    """
    f = 0.
    for j, c in enumerate(parameters):
        f += pow(c,4) * (j+1.0) + random.random()
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))
    return f


#------------------------------------------
def DeJongStep(parameters, fileprint=True):
    """
    De Jong's step function:
    The third De Jong function, Equation (19) of [2]
    minimum is f(x)=0.0 at xi=-5-n where n=[0.0,0.12]
    """
    f = 6*len(parameters)
    for c in parameters:
        if abs(c) <= 5.12:
            f += floor(c)
        elif c > 5.12:
            f += 30 * (c - 5.12)
        else:
            f += 30 * (5.12 - c)
    if fileprint:
        with open("dumpfile.dat", "a") as outfile:
            outstr = ' '.join([str(x) for x in parameters])
            outfile.write('%s | %s \n'%(outstr, f))
    return f
#------------------------------------------


if __name__ == "__main__":
    parameter = [1.0, 1.0, 1.0, 1.0]
    print(Colville(parameter))
