def expandhead_radiallatin(npoints, ndim, tree, rmax):
    seedpoints = init_latin_hypercube_sampling(np.array([0.0]), np.array([rmax]), npoints)
    for rcur in seedpoints:
        u = np.random.normal(0.0, 1.0, ndim)  # an array of d normally distributed random variables
        norm = np.sum(u**2) **(0.5)
        x = rcur*u/norm
        nodedata = indata.newdataobject()
        nodedata.setstructure(x)
        tree.expandfromdata(newdata=nodedata)


