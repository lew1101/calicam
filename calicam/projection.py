import numpy as np
import scipy.sparse.linalg 
from math import sqrt

from collections.abc import Iterator, Iterable
from .types import *

def construct_A(world_coords: list[Vec3], image_coords: list[Vec2]) -> np.ndarray:
    rows = []
    for (x, y, z), (u, v) in zip(world_coords, image_coords):
        rows.append([x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -u*x, -u*y, -u*z, -u])
        rows.append([0.0, 0.0, 0.0, 0.0, x, y, z, 1.0, -v*x, -v*y, -v*z, -v])
    return np.array(rows)

    
def generate_proj_matrix(world_coords: list[Vec3], image_coords: list[Vec2]) -> tuple[float, ProjMatrix]:
    A = construct_A(world_coords, image_coords)
    M = np.matmul(np.transpose(A), A)
    
    eigval, p = scipy.sparse.linalg.eigs(M, k=1, which='SM')
    proj_matrix = p.real.reshape(3, 4)
    
    return eigval, proj_matrix


def project(projection_matrix: ProjMatrix, world_coords: Vec3 | Iterable[Vec3]) -> Vec2 | list[Vec2]:
    def project_impl(wc: Vec3) -> Vec2:
        ut, vt, wt = np.dot(projection_matrix, (*wc, 1))
        return (ut/wt, vt/wt)

    if not isinstance(world_coords, Iterable):  
        return project_impl(world_coords)
    else:
        return [project_impl(world_coord) for world_coord in world_coords]
    
    
def calculate_reproj_error(actual_coords: Vec2 | Iterator[Vec2], reproj_coords: Vec2 | Iterator[Vec2]):
    def error_impl(ac: Vec2, rc: Vec2) -> float:
        return sqrt((ac[0]-rc[0])**2 + (ac[1]-rc[1])**2)
    
    if not isinstance(actual_coords, Iterable) and not isinstance(reproj_coords, Iterable):
        return error_impl(actual_coords, reproj_coords)
    else:
        errors = [error_impl(ac, rc) for ac, rc in zip(actual_coords, reproj_coords)]
        avg_error = sum(errors) / len(errors)
        return errors, avg_error
