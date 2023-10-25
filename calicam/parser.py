import csv

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


def parse_csv(path: str) -> tuple[list[Vec3], list[Vec2]]:
    """
    Parses a csv and returns a list of 3D scene points and their corresonding
    2D image mappings.
    CSV format: x,y,z,u,v
    where 3D point = (x, y, z) and 2D point (u,v)
    """
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")

        world_coords = []
        image_coords = []
        for lno, line in enumerate(reader, start=1):
            assert len(line) == 5, f"Data on line {lno} in {path} is invalid."

            x, y, z, u, v = (float(s) for s in line)

            world_coords.append((x, y, z))
            image_coords.append((u, v))

    return world_coords, image_coords
