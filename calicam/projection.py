import numpy as np
import scipy.sparse.linalg
from nptyping import Shape, Double

from .vecs import *

ProjMatrix = np.ndarray[Shape["3, 4"], Double]


def generate_estimation_matrix(world_coords: list[Vec3f], image_coords: list[Vec2f]) -> np.ndarray:
    """
    Generates an estimation matrix from list of 3D world coords and
    their corresponding pixel coord mappings
    """
    rows = []
    for (x, y, z), (u, v) in zip(world_coords, image_coords):
        rows.append([x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z, -u])
        rows.append([0.0, 0.0, 0.0, 0.0, x, y, z, 1.0, -v * x, -v * y, -v * z, -v])
    return np.array(rows)


def generate_proj_matrix(world_coords: list[Vec3f], image_coords: list[Vec2f]) -> tuple[ProjMatrix, float]:
    """
    Takes 3D calibration points their corresponding pixel coord mappings and
    returns the projection matrix as a 3x4 matrix
    """
    assert len(world_coords) == len(image_coords), \
        f"The number of world coordinates ({world_coords}) and image coordinates ({image_coords}) do not match."

    assert len(world_coords) >= 6, \
        f"Need at least 6 calibration points, but only {len(world_coords)} were provided."

    G = generate_estimation_matrix(world_coords, image_coords)
    M = G.T @ G

    eigval, p = scipy.sparse.linalg.eigs(M, k=1, which="SM")  # solve for minimum p using eigenvalue problem
    proj_matrix = p.real.reshape(3, 4)  # take only real part of p and convert into 3x4 matrix

    return proj_matrix, eigval


def project_point(projection_matrix: ProjMatrix, world_coords: Vec3f) -> Vec2f:
    """
    Calculate pixel coordinate from 3D world coord using projection matrix
    """
    im_point = projection_matrix @ to_homogenous(world_coords)
    return to_inhomogenous(im_point)  # turn into inhomogenous coords
