import csv

from .types import Vec2, Vec3

def parse_csv(path: str) -> tuple[list[Vec3], list[Vec2]]:
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=",")

        world_coords = []
        image_coords = []
        for lno, line in enumerate(reader, start=1):
            if len(line) != 5:
                raise Exception(f"Data on line {lno} in {path} is invalid.")
            
            x, y, z, u, v = (float(s) for s in line)
            
            world_coords.append((x, y, z))
            image_coords.append((u, v))
            
    return world_coords, image_coords

