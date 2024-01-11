from math import sqrt

Vecf = tuple[float, ...]

Vec2f = tuple[float, float]
Vec3f = tuple[float, float, float]


def to_homogenous(vec: Vecf) -> Vecf:
    return (*vec, 1.0)


def to_inhomogenous(vec: Vecf) -> Vecf:
    return tuple(map(lambda v_i: v_i / vec[-1], vec[:-1]))


def euclidean(a: Vecf, b: Vecf) -> float:
    assert len(a) == len(b), f"Vectors need to have the same dimension"
    return sqrt(sum((b_i - a_i)**2 for a_i, b_i in zip(a, b)))
