from dataclasses import dataclass
from nptyping import NDArray, Shape, Double

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

ProjMatrix = NDArray[Shape["3, 4"], Double]
@dataclass
class Parameters:
    focal_lengths: Vec2
    principal_point: Vec2
    angles: Vec3
    translation_vector: Vec3
