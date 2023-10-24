#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

import calicam


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("path", help="Path to csv file")
    parser.add_argument("-d", "--data", help="Path to csv file")
    parser.add_argument("-g", "--graph", nargs='?',
                        default=None, const="", help="Generate graph")

    try:
        np.set_printoptions(precision=4)

        args = parser.parse_args()

        calibration_world_coords, calibration_image_coords = calicam.parse_csv(
            args.path)
        _, proj_matrix = calicam.generate_proj_matrix(
            calibration_world_coords, calibration_image_coords)

        K, R, t = calicam.decompose_proj_matrix(proj_matrix)
        
        (cx, cy), (fx, fy) = calicam.extract_intrinsics(K)
        a, b, g = calicam.extract_orientation_zyx(R)
        tx, ty, tz = t

        print('\n', '\n\n'.join((
            f"Projection Matrix: \n{proj_matrix}",
            f"Focal Lengths: \n\tf_x = {fx:.2f} px \n\tf_y = {fy:.2f} px",
            f"Principal Point: \n\tc_x = {cx:.2f} px \n\tc_y = {cy:.2f} px",
            f"Translation: \n\tt = [{tx:.2f}, {ty:.2f}, {tz:.2f}] mm",
            f"Orientation: \n\t\u03B1 = {a:.2f}° \n\t\u03b2 = {b:.2f}° \n\t\u03B3 = {g:.2f}°",
        )))

        if args.data:
            data_path = args.data

            data_world_coords, data_image_coords = calicam.parse_csv(data_path)
            reprojections = calicam.project(proj_matrix, data_world_coords)
            errors, avg_err = calicam.calculate_reproj_error(
                data_image_coords, reprojections)
            max_err = max(errors)

            print(
                f"\nReprojection Errors: \n\t\u03BC_max = {max_err:.3f} px \n\t\u03BC_avg = {avg_err:.3f} px",)

        if args.graph != None:
            image_path = args.graph

            if not args.data:
                raise Exception(
                    "Path to data csv file must be provided using -d flag to produce graph")

            _, ax = plt.subplots(figsize=(8, 10))

            plt.gca().invert_yaxis()

            if image_path != "":
                img = plt.imread(image_path)
                ax.imshow(img)
                ax.autoscale(False)

            ax.scatter(cx, cy, label='Principal Point', s=60, c='crimson')
            ax.scatter(*zip(*reprojections), label='Reprojected Points',
                    s=40, c="yellow", alpha=0.8)
            ax.scatter(*zip(*calibration_image_coords),
                    label='Calibration Points', s=60, c="blue", alpha=0.8)

            plt.gca().update({
                "title": image_path,
                "xlabel": '$u$ (px)',
                "ylabel": '$v$ (px)'})
            plt.legend()
            plt.show()

        return 0
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    main()
