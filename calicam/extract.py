import numpy as np
import scipy.linalg 
from math import sqrt

from .types import *

def decompose_proj_matrix(proj_matrix: ProjMatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K, R = scipy.linalg.rq(proj_matrix[:,:3])
    D = np.diag(np.sign(np.diag(K)))
    
    if np.linalg.det(D) < 0:
        D[1,1] *= -1
        
    K = K @ D
    R = D @ R
    t = np.linalg.inv(K) @ proj_matrix[:,3]

    return K, R, t

def extract_focal_length(calibration: np.ndarray) -> Vec2:
    return (calibration[0][0], calibration[1][1])

def extract_principal_point(calibration: np.ndarray) -> Vec2:
    return(calibration[0][2], calibration[1][2])

def extract_orientation(rotation: np.ndarray) -> Vec3:
    return (np.degrees(np.arcsin(-rotation[2][0])), 
              np.degrees(np.arcsin(rotation[1][0]/sqrt(1-(rotation[2][0])**2))), 
              np.degrees(np.arcsin(rotation[2][1]/sqrt(1-(rotation[2][0])**2))))

def extract_parameters(proj_matrix: ProjMatrix) -> Parameters:
    K, R, t = decompose_proj_matrix(proj_matrix)
    focal_lengths = extract_focal_length(K)
    principal_point = extract_principal_point(K)
    angles = extract_orientation(R)
    t_vec = tuple([n for n in t])
    print(K, R)
    return Parameters(focal_lengths, principal_point, angles, t_vec)
