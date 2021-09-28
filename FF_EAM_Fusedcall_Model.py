from NetSetUp import getmodel
import numpy as np
import tensorflow as tf
import pickle
from utils import pretty_print

#================================================================================
class ForcefieldModel():
    #---------------------------------------------------------------------------
    def __init__(self, nlayers=None, symbasis=None, rcut=8.0, offset=0.0 ):
        self.nSymLayers = {'pair':len(symbasis['pair']),
                           'density':len(symbasis['density']),
                           'embedding':len(symbasis['embedding'])}
        self.actfunc = symbasis
        self.pairsym = getmodel(nlayer=self.nSymLayers['pair'], inputdim=1, actfunc=symbasis['pair'])
        self.denssym = getmodel(nlayer=self.nSymLayers['density'], inputdim=1, actfunc=symbasis['density'])
        self.embsym = getmodel(nlayer=self.nSymLayers['embedding'], inputdim=1, actfunc=symbasis['embedding'])
        self.rcut = rcut
        self.rcutsq = rcut*rcut
        self.offset = tf.Variable(offset)
#        self.offset = tf.constant(offset)

    #----------------------------------------------------------------------------
    def __call__(self, npairlist, natomslist, atomwisepairlist, fuseddistlist):

        # For PAIRWISE energy term
        fusedresults = self.pairsym(fuseddistlist)
        if tf.reduce_any(tf.math.is_nan(fusedresults)) or tf.reduce_any(tf.math.is_inf(fusedresults)):
#            print('Error in Pairsym call; outpair ', fusedresults)
            return tf.constant([np.nan]*len(fusedresults))
        
        # For ATOMWISE Density term
        fused_dens_results = self.denssym(fuseddistlist)
        if tf.reduce_any(tf.math.is_nan(fused_dens_results)) or tf.reduce_any(tf.math.is_inf(fused_dens_results)):
#            print('Error in Denssym call; outdens ', fused_dens_results)
            return tf.constant([np.nan]*len(fused_dens_results))        

        nlow = 0
        nhigh = 0
        
        pairenergy = []
        atom_densities = []
        for i, npair in enumerate(npairlist):
            nlow = nhigh
            nhigh += npair
            pairenergy.append(tf.math.reduce_sum(fusedresults[nlow:nhigh])/natomslist[i])
            
            # Collecting Atomic density outputs for each atom
            i_relevant_tensor = fused_dens_results[nlow:nhigh] # Tensor with pair terms of Structure i
            inner_list = atomwisepairlist[i]                   # List of list of neighbor pairs for each atom in struct i
            for j, atom in enumerate(inner_list):
                atom_densities.append(tf.math.reduce_sum(tf.gather_nd(i_relevant_tensor, [[idx] for idx in atom])))
                
        pairenergy = tf.stack(pairenergy,axis=0)


        # For EMBEDDING Density term
        atom_densities = tf.expand_dims(tf.stack(atom_densities,axis=0),axis=1)
        fused_emb_results = self.embsym(atom_densities)
        
        if tf.reduce_any(tf.math.is_nan(fused_emb_results)) or tf.reduce_any(tf.math.is_inf(fused_emb_results)):
#            print('Error in Embsym call; outemb ', fused_emb_results)
            return tf.constant([np.nan]*len(fused_emb_results))        

        embenergy = []
        for i, natom in enumerate(natomslist):
            nlow = nhigh
            nhigh += natom
            embenergy.append(tf.math.reduce_sum(fused_emb_results[nlow:nhigh])/natomslist[i])
        embenergy = tf.stack(embenergy,axis=0)


        # Total Energy
        totalenergy = tf.math.add(pairenergy,embenergy)
        if tf.reduce_any(tf.math.is_nan(totalenergy)) or tf.reduce_any(tf.math.is_inf(totalenergy)):
#            print('Error in Total energy call; totalenergy', totalenergy )
            return tf.constant([np.nan]*len(totalenergy))

        totalenergy = tf.math.add(totalenergy, self.offset)

        return totalenergy
    #----------------------------------------------------------------------------
    def buildneighlist(self, atompos):
        natoms = atompos.shape[0]
        neighlist = [list() for iatom in range(natoms)]
        for iAtom, atom in enumerate(atompos):
            if iAtom+1 == natoms:
                break
            distarr = atompos[iAtom+1:] - atom
            distarr = np.square(distarr)
            distarr = np.sum(distarr, axis=1)
            neighs = np.where(distarr <= self.rcutsq)
            neighs = neighs[0] + iAtom+1
            for val in neighs:
                if iAtom == val:
                    continue
                if val not in neighlist[iAtom]:
                    neighlist[iAtom].append(val)
                if iAtom not in neighlist[val]:
                    neighlist[val].append(iAtom)
        for iAtom, neighs in enumerate(neighlist):
            neighs.sort()
        return neighlist
    #----------------------------------------------------------------------------
    # p(A1) = p(r12) + p(13) + p(14).... p(1N)
    # p(A2) = p(r12) + p(23) + p(24).... p(2N)
    def builddistlist(self, atompos, neighlist):
        natoms = atompos.shape[0]
        outpairs = []
        for iAtom, iCoords in enumerate(atompos):
            if len(neighlist[iAtom]) < 1:
                continue
            for jAtom in neighlist[iAtom]:
                if iAtom <= jAtom:
                    continue
                vij = atompos[jAtom] - atompos[iAtom]
                rij = np.linalg.norm(vij)
#                rij_inv = 1.0/rij
#                rij_exp = np.exp(-3.0*rij)
                #print(rij, rij_inv, rij_exp)
                outpairs.append([rij])
                #outpairs.append([rij, rij_inv, rij_exp])
        outpairs = np.array(outpairs, dtype=np.float32)
        return outpairs

    #----------------------------------------------------------------------------
    def buildatomwisedistlist(self, atompos, neighlist):
        natoms = atompos.shape[0]
        atomwisedistlist = [list() for iatom in range(natoms)]
        outpairs = []
        for iAtom, iCoords in enumerate(atompos):
            #if len(neighlist[iAtom]) < 1:
                #continue
            assert len(neighlist[iAtom]) >= 1, "Empty Neighbor list."
            for jAtom in neighlist[iAtom]:
                vij = atompos[jAtom] - atompos[iAtom]
                rij = np.linalg.norm(vij)
                atomwisedistlist[iAtom].append([rij])
            atomwisedistlist[iAtom] = np.array(atomwisedistlist[iAtom], dtype=np.float32)
        return atomwisedistlist
    #----------------------------------------------------------------------------
    def get_weights(self):
        pairweights = self.pairsym.get_weights()
        densweights = self.denssym.get_weights()
        embweights = self.embsym.get_weights()
        outweights = pairweights + densweights + embweights + [self.offset]
        return outweights
    #----------------------------------------------------------------------------
    def get_npweights(self):
        pairweights = self.pairsym.get_npweights()
        densweights = self.denssym.get_npweights()
        embweights = self.embsym.get_npweights()
        outweights = pairweights + densweights + embweights
        return outweights
    #----------------------------------------------------------------------------
    def set_npweights(self, inweights):
        pairweights = inweights[:self.nSymLayers['pair']+1]
        densweights = inweights[self.nSymLayers['pair']+1:self.nSymLayers['pair']+self.nSymLayers['density']+2]
        embweights = inweights[self.nSymLayers['pair']+self.nSymLayers['density']+2:]
        self.pairsym.set_npweights(pairweights)
        self.denssym.set_npweights(densweights)
        self.embsym.set_npweights(embweights)
    #----------------------------------------------------------------------------
    def mask_weights(self, threshold):
        W = self.pairsym.get_npweights()
        for w in W:
            w[np.abs(w) < threshold] = 0
        self.pairsym.set_npweights(W)

        W = self.denssym.get_npweights()
        for w in W:
            w[np.abs(w) < threshold] = 0
        self.denssym.set_npweights(W)

        W = self.embsym.get_npweights()
        for w in W:
            w[np.abs(w) < threshold] = 0
        self.embsym.set_npweights(W)
    #----------------------------------------------------------------------------
    def save_model(self, filename='symmodel.ml'):
        savelist = [self.nSymLayers, self.actfunc, self.rcut,
                     self.pairsym.get_npweights(), self.denssym.get_npweights(),
                     self.embsym.get_npweights()]

        with open(filename, 'wb') as outf:
            pickle.dump(savelist, outf)
    #----------------------------------------------------------------------------
    def load_model(self, filename='symmodel.ml'):
        savelist = pickle.load( open( filename, "rb" ) )

        self.nSymLayers = savelist[0]
        self.actfunc = savelist[1]
        self.rcut = savelist[2]
        self.rcutsq = self.rcut*self.rcut
        self.pairsym = getmodel(nlayer=self.nSymLayers['pair'], inputdim=1, actfunc=self.actfunc['pair'])
        self.pairsym.set_npweights(savelist[3])

        self.denssym = getmodel(nlayer=self.nSymLayers['density'], inputdim=1, actfunc=self.actfunc['density'])
        self.denssym.set_npweights(savelist[4])

        self.embsym = getmodel(nlayer=self.nSymLayers['embedding'], inputdim=1, actfunc=self.actfunc['embedding'])
        self.embsym.set_npweights(savelist[5])
    #----------------------------------------------------------------------------
    def pretty_output(self, threshold=0):
        pair_term = pretty_print.hetrogeneous_network(self.pairsym.get_npweights(), self.actfunc['pair'],
                                                     ['r'], threshold=threshold)
        density_term = pretty_print.hetrogeneous_network(self.denssym.get_npweights(), self.actfunc['density'],
                                                     ['r'], threshold=threshold)
        emb_term = pretty_print.hetrogeneous_network(self.embsym.get_npweights(), self.actfunc['embedding'],
                                                     ['r'], threshold=threshold)
        print('---------------------------------')
        print('Pair term:\n', pair_term)
        print('---------------------------------')
        print('Density term:\n', density_term)
        print('---------------------------------')
        print('Emb. term:\n', emb_term)
        print('---------------------------------')
        return pair_term, density_term, emb_term
        


