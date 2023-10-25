import numpy as np
import scipy.sparse.linalg
from math import sqrt

from nptyping import NDArray, Shape, Double
from collections.abc import Iterable

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

ProjMatrix = NDArray[Shape["3, 4"], Double]


def generate_input_matrix(world_coords: list[Vec3], image_coords: list[Vec2]) -> np.ndarray:
    """
    Generates a calibration matrix from list of 3D world coords and
    their corresponding pixel coord mappings
    """
    rows = []
    for (x, y, z), (u, v) in zip(world_coords, image_coords):
        rows.append([x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z, -u])
        rows.append([0.0, 0.0, 0.0, 0.0, x, y, z, 1.0, -v * x, -v * y, -v * z, -v])
    return np.array(rows)


def generate_proj_matrix(
    world_coords: list[Vec3], image_coords: list[Vec2]
) -> tuple[ProjMatrix, float]:
    """
    Takes 3D calibration points their corresponding pixel coord mappings and
    returns the projection matrix as a 3x4 matrix
    """
    assert len(world_coords) == len(image_coords), (
        f"The number of world coordinates ({world_coords}) "
        "and image coordinates ({image_coords}) do not match."
    )
    assert (
        len(world_coords) >= 6
    ), f"Need at least 6 calibration points, but only {len(world_coords)} were provided."

    A = generate_input_matrix(world_coords, image_coords)
    M = A.T @ A

    eigval, p = scipy.sparse.linalg.eigs(M, k=1, which="SM")
    proj_matrix = p.real.reshape(3, 4)

    return proj_matrix, eigval


def project(projection_matrix: ProjMatrix, world_coords: Vec3) -> Vec2:
    """
    Calculate pixel coordinate from 3D world coord using projection matrix
    """
    u, v, w = projection_matrix @ (*world_coords, 1.0)
    return (u / w, v / w)


def calculate_reproj_error(actual: Vec2, reprojected: Vec2) -> float:
    """
    Calculate reprojection error using L2 norm (euclidean distance)
    """
    return sqrt((reprojected[0] - actual[0]) ** 2 + (reprojected[1] - actual[1]) ** 2)
