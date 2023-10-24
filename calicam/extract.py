import numpy as np
import scipy.linalg
from math import sqrt
from dataclasses import dataclass

from .projection import ProjMatrix


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


@dataclass
class Parameters:
    focal_lengths: Vec2
    principal_point: Vec2
    angles: Vec3
    translation_vector: Vec3


def decompose_proj_matrix(proj_matrix: ProjMatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K, R = scipy.linalg.rq(proj_matrix[:, :3])

    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R

    return K, R


def extract_focal_length(calibration: np.ndarray) -> Vec2:
    return (calibration[0][0], calibration[1][1])


def extract_principal_point(calibration: np.ndarray) -> Vec2:
    return (calibration[0][2], calibration[1][2])


def extract_orientation(rotation: np.ndarray) -> Vec3:
    return (np.degrees(np.arcsin(-rotation[2][0])),
            np.degrees(np.arcsin(rotation[1][0]/sqrt(1-(rotation[2][0])**2))),
            np.degrees(np.arcsin(rotation[2][1]/sqrt(1-(rotation[2][0])**2))))


def extract_parameters(proj_matrix: ProjMatrix) -> Parameters:
    K, R = decompose_proj_matrix(proj_matrix)
    Ks = K / K[2][2]

    focal_lengths = extract_focal_length(Ks)
    principal_point = extract_principal_point(Ks)
    angles = extract_orientation(R)
    t_vec = tuple(-np.linalg.inv(proj_matrix[:, :3]) @ proj_matrix[:, 3])
    return Parameters(focal_lengths, principal_point, angles, t_vec)
