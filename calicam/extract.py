import numpy as np
import scipy.linalg
from math import sqrt
from nptyping import NDArray, Shape, Double

from .projection import ProjMatrix


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

CalMatrix = NDArray[Shape["3, 3"], Double]
RotMatrix = NDArray[Shape["3, 3"], Double]


def decompose_proj_matrix(
    proj_matrix: ProjMatrix,
) -> tuple[CalMatrix, RotMatrix, Vec3]:
    """
    Decomposes the projection matrix into the calibration matrix,
    rotation matrix, and translation matrix.
    """
    K, R = scipy.linalg.rq(proj_matrix[:, :3])

    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R

    t = tuple(-np.linalg.inv(proj_matrix[:, :3]) @ proj_matrix[:, 3])

    return K, R, t


def extract_intrinsics(K: CalMatrix) -> tuple[Vec2, Vec2]:
    """
    Extract principle point and focal lengths from calibration matrix
    """
    Ks = K / K[2][2]

    focal_lengths = (Ks[0][0], Ks[1][1])
    principal_point = (Ks[0][2], Ks[1][2])
    return principal_point, focal_lengths


def extract_orientation_zyx(R: RotMatrix) -> Vec3:
    """
    Extract tait-bryan angles (zyx) from rotation matrix
    """
    return (
        np.degrees(np.arcsin(-R[2][0])),
        np.degrees(np.arcsin(R[1][0] / sqrt(1 - (R[2][0]) ** 2))),
        np.degrees(np.arcsin(R[2][1] / sqrt(1 - (R[2][0]) ** 2))),
    )
