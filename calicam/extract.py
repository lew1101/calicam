import numpy as np
import scipy.linalg
from math import sqrt
from nptyping import  Shape, Double

from .projection import generate_proj_matrix, ProjMatrix
from .vecs import *

CalMatrix = np.ndarray[Shape["3, 3"], Double]
RotMatrix = np.ndarray[Shape["3, 3"], Double]


def calibrate_camera(world_coords: list[Vec3f],
                     image_coords: list[Vec2f]) -> tuple[ProjMatrix, CalMatrix, RotMatrix, Vec3f]:
    """
    Decomposes the projection matrix into the calibration matrix,
    rotation matrix, and translation matrix.
    """
    proj_matrix, _ = generate_proj_matrix(world_coords, image_coords)

    K, R = scipy.linalg.rq(proj_matrix[:, :3])  # rq decomposition

    # enforce positive diagonal on K
    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R

    # scale projection matrix and calibration matrix so that
    scale_factor = 1 / K[2][2]
    proj_matrix *= scale_factor
    K *= scale_factor

    # extract translation vector from P
    t = tuple(-np.linalg.inv(proj_matrix[:, :3]) @ proj_matrix[:, 3])

    return proj_matrix, K, R, t


def extract_intrinsics(K: CalMatrix) -> tuple[Vec2f, Vec2f]:
    """
    Extract principle point and focal lengths from calibration matrix
    """
    focal_lengths = (K[0][0], K[1][1])
    principal_point = (K[0][2], K[1][2])
    return principal_point, focal_lengths


def extract_orientation_zyx(R: RotMatrix) -> Vec3f:
    """
    Extract tait-bryan angles (zyx) from rotation matrix
    """
    return (
        np.degrees(np.arcsin(R[2][1] / sqrt(1 - (R[2][0])**2))),  # alpha (x rotation)
        np.degrees(np.arcsin(-R[2][0])),  # beta (y rotation)
        np.degrees(np.arcsin(R[1][0] / sqrt(1 - (R[2][0])**2))),  # gamma (z rotation)
    )
