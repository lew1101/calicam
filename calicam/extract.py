import numpy as np
import scipy.linalg 
from math import sqrt

from .types import *

def decompose_proj_matrix(proj_matrix: ProjMatrix) -> tuple[np.ndarray, np.ndarray]:
    calibration, rotation = scipy.linalg.rq(proj_matrix[:,:-1], mode='economic')
    print(proj_matrix[:,:-1])
    print('\n')
    print(rotation)
    print('\n')
    print(calibration)
    return calibration, rotation


def extract_parameters(calibration: np.ndarray, rotation: np.ndarray) -> Parameters:
    focal_lengths = (calibration[0][0]/calibration[2][2], calibration[1][1]/calibration[2][2])
    principal_point = (calibration[0][2]/calibration[2][2], calibration[1][2]/calibration[2][2])
    angles = (np.degrees(np.arcsin(-rotation[2][0])), 
              np.degrees(np.arcsin(rotation[1][0]/sqrt(1-(rotation[2][0])**2))), 
              np.degrees(np.arcsin(rotation[2][1]/sqrt(1-(rotation[2][0])**2))))
    
    return Parameters(focal_lengths, principal_point, angles)
