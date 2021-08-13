from __future__ import division, print_function
import numpy as n
import itertools as it
from time import time


def find_accessible_rlvs(crystal, wavelength):
    """
    Generates a list of accessible reciprocal lattice
    vectors. To be accessible, the magnitude of a rlv's
    wavevector must be less than twice that of the input
    radiation's wavenumber."""
    
    # The wavenumber of the input wavelength
    nu = 2*n.pi/wavelength
    
    # Now we find the shortest distance to a wall of a 
    # parallelogram "shell" in the reciprocal lattice
    min_step = min(abs(n.dot(
        (crystal.rlattice[0]+crystal.rlattice[1]
         +crystal.rlattice[2]),
        n.cross(crystal.rlattice[i],crystal.rlattice[j])
        /n.linalg.norm(n.cross(crystal.rlattice[i],crystal.rlattice[j]))))
                   for i,j in [(0,1),(1,2),(2,0)])
    
    # If we look at all the points in this many parallelogram
    # "shells", we can't miss all the accessible wavevectors
    num_shells = int(2*nu / min_step)
    
    # Now we generate these possibilities
    possibilities = [(crystal.rlattice[0]*h + crystal.rlattice[1]*j
                      + crystal.rlattice[2]*k)
                     for h,j,k in it.product(
                             range(-num_shells,num_shells+1),
                             repeat=3)]
    
    # And we filter the possibilities, getting rid of all the
    # rlvs that are too long and the 0 vector
    rlvs = [rlv for rlv in possibilities 
            if n.linalg.norm(rlv) < 2*nu
            and not n.allclose(rlv,0)]
    
    return n.array(rlvs)


def powder_XRD(crystal, wavelength, get_mults=False):
    """
    Generates a powder XRD spectrum for radiation with the
    given wavelength (in angstroms)
    """
    
    # The wavenumber of the input wavelength
    nu = 2*n.pi/wavelength

    # Make a list of the accessible rlvs
    rlvs = find_accessible_rlvs(crystal,wavelength)
    
    # Now we calculate the scattering intensity from each rlv
    intensities = {
        tuple(rlv): n.abs(crystal.structure_factor(rlv))**2
        for rlv in rlvs}
    
    # Now sum up all rlvs with the same magnitude. We also
    # get rid of all the scattering vectors with 0 intensity
    magnitudes = {}
    multiplicities = {}
    for rlv, intensity in intensities.items():
        repeat = False
        mag = n.linalg.norm(rlv)
        for oldmag in magnitudes:
            if n.isclose(mag,oldmag):
                magnitudes[oldmag] += intensity
                multiplicities[oldmag] += 1
                repeat = True
                break
        if not repeat and not n.isclose(mag,0):
            multiplicities[mag] = 1
            magnitudes[mag] = intensity
        
    # Now we reformat the multiplicity data in a nice way
    multiplicities = {2 * n.arcsin(mag / (2 * nu)) * 180 / n.pi:
                      multiplicity
                      for mag, multiplicity in multiplicities.items()
                      if not n.allclose(magnitudes[mag],0)}

    # And now we calculate the scattering intensities
    # (a.u. per steradian) as a function of scattering angle
    intensities = {2 * n.arcsin(mag / (2 * nu)) * 180 / n.pi:
                   intensity * 
                   # This factor corrects for the fact that the same total
                   # power in the debye scherrer rings is more
                   # concentrated when 2\theta is near 0 or 2pi
                   1 / n.sin(2*n.arcsin(mag/(2*nu))) *
                   # This factor corrects for the probability that any
                   # given crystal domain will scatter into the rlv
                   1 / mag *
                   # This factor corrects for polarization effects,
                   # Assuming an unpolarized input beam and no polarization
                   # analysis
                   (1 + n.cos(2*n.arcsin(mag/(2*nu)))**2)/2
                   for mag, intensity in magnitudes.items()
                   if not n.allclose(intensity,0)}
    if get_mults:
        return intensities, multiplicities
    else:
        return intensities



def spectrumify(scattering_data, instr_broadening=0.1, start_angle=0, end_angle=180, step=0.02):
    """
    This is just a nice function to turn the raw scattering data
    into a human-readable scattering spectrum
    """
    num_points = int((end_angle - start_angle)/step)
    graph_angles = n.linspace(start_angle,end_angle,num_points)
    graph_intensities = n.zeros(graph_angles.shape)
    
    for angle, intensity in sorted(scattering_data.items()):
        graph_intensities += intensity * \
                             n.exp(-(graph_angles - angle)**2 / \
                                   (2*(instr_broadening)**2))
        
    return graph_angles, graph_intensities

# https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
# constant - https://towardsdatascience.com/4-must-know-features-of-python-dictionaries-d62af8c22fd2
# https://en.wikipedia.org/wiki/Lattice_constant
BCC_Element_list = {"Li": 3.49, "Na": 4.23, "K": 5.23, "Fe": 2.856, \
                    "Cr": 2.88, "V": 3.02, "Nb": 3.30, "Mo": 3.155, \
                    "Ta" : 3.31, "W": 3.155, "Cs": 6.05, "Ba": 5.02, "Rb": 5.59}
FCC_Element_list = {"Al": 4.046, "Ni": 3.499, "Cu": 3.597, \
                    "Pd": 3.859, "Ag": 4.079, \
                    "Pt": 3.912, "Au": 4.065, "Pb": 4.920, "Ar": 5.26, "Ca": 5.58, "Ce": 5.16, \
                   "Xe": 6.20, "Yb": 5.49, "Ir": 3.84, "Ac": 5.31, "Th": 5.08}

Hexagonal_Element_list = {"Mg": [3.21,1.624], "Ti": [2.95,1.588], "Co": [2.51,1.622], "Zn": [2.66,1.856], \
                    "Zr": [3.23,1.593], "Cd":[2.98,1.886], "La": [3.75,1.619], "Nd": [3.66,1.614], \
                    "Gd": [3.64,1.588], "Y": [3.65,1.571], "Hf": [3.20, 1.582]}
Orthorhombic_Element_list = {"FeOOH":0}

Goethite = {'name':"FeOOH", 'latice':"Orthorhombic", 'l_const':[0.9956,0.30215,0.4608],
           'basis': [('Fe',[0.145,1/4,-0.045]),('O',[-0.199,1/4,0.288]),('O',[-0.053,1/4,-0.198]),('H',[-0.08,1/4,-0.38])]}

# http://pd.chem.ucl.ac.uk/pdnn/inst1/anode.htm
XRD_radiation_list = {'Cu': 1.5405, 'Mo': 0.709319, 'Co': 1.789010, 'Cr':2.289760, 'Ag': 0.5594075 , 'Fe': 1.94}



