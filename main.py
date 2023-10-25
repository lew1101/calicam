#!/usr/bin/env python3
import os
import sys

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import numpy as np
import matplotlib.pyplot as plt

import calicam


def main() -> int:
    parser = ArgumentParser(
        prog="calicam",
        description=(
            "Generates projection matrix and calculates intrinsic and extrinsic parameters.\n"
            "CSV inputs are in the format: x,y,z,u,v where 3D point = (x, y, z) and 2D point (u,v)"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    parser.add_argument("path", help="path to csv file with calibration points")
    parser.add_argument("-d", "--data", metavar="PATH", help="path to csv file with model verification data")
    parser.add_argument("-g", "--graph", nargs="?", const="", metavar="IMAGE", help="generate graph")
    parser.add_argument("-o", "--out", metavar="PATH", help="graph output location")

    args = parser.parse_args()

    try:
        csv_path: str = args.path
        
        cali_world_coords, cali_image_coords = calicam.parse_csv(csv_path)
        proj_matrix, _ = calicam.generate_proj_matrix(
            cali_world_coords, cali_image_coords
        )

        cali_matrix, rot_matrix, t = calicam.decompose_proj_matrix(proj_matrix)

        (cx, cy), (fx, fy) = calicam.extract_intrinsics(cali_matrix)
        a, b, g = calicam.extract_orientation_zyx(rot_matrix)
        tx, ty, tz = t

        np.set_printoptions(precision=4)

        print(
            "\n",
            "\n\n".join(
                (
                    f"Projection Matrix: \n{proj_matrix}",
                    f"Calibration Matrix: \n{cali_matrix}",
                    f"Rotation Matrix: \n{rot_matrix}",
                    f"Focal Lengths: \n\tf_x = {fx:.2f} px \n\tf_y = {fy:.2f} px",
                    f"Principal Point: \n\tc_x = {cx:.2f} px \n\tc_y = {cy:.2f} px",
                    f"Translation: \n\tt_x = {tx:.2f} mm \n\tt_y = {ty:.2f} mm \n\tt_z = {tz:.2f} mm",
                    f"Orientation: \n\t\u03B1 = {a:.2f}° \n\t\u03B2 = {b:.2f}° \n\t\u03B3 = {g:.2f}°",
                )
            ),
        )

        if args.data is not None:
            data_path: str = args.data

            assert os.path.isfile(data_path), f"{data_path} does not exist."
            assert data_path.endswith(
                ".csv"
            ), f'{data_path} does not end with the extension ".csv".'

            data_world_coords, data_image_coords = calicam.parse_csv(data_path)
            reprojections = [
                calicam.project(proj_matrix, world_coord)
                for world_coord in data_world_coords
            ]
            reproj_errors = [
                calicam.calculate_reproj_error(actual_coord, reproj_coord)
                for actual_coord, reproj_coord in zip(data_image_coords, reprojections)
            ]
            max_err = max(reproj_errors)
            avg_err = sum(reproj_errors) / len(reproj_errors)

            print(
                f"\nReprojection Errors: \n\t\u03BC_max = {max_err:.3f} px \n\t\u03BC_avg = {avg_err:.3f} px",
            )

        if args.graph is not None:
            image_path: str = args.graph

            _, ax = plt.subplots(figsize=(8, 10))

            plt.gca().invert_yaxis()

            if image_path != "":
                assert os.path.isfile(image_path), f"{image_path} does not exist."
                assert args.data, (
                    "Path to data csv file must be provided"
                    "using -d flag to produce graph."
                )

                img = plt.imread(image_path)
                ax.imshow(img)
                ax.autoscale(False)

            ax.scatter(
                *zip(*data_image_coords),
                label="Data Points",
                s=50,
                c="deeppink",
                alpha=0.7,
            )
            ax.scatter(
                *zip(*reprojections),
                label="Reprojected Points",
                s=50,
                c="yellow",
                alpha=0.7,
            )
            ax.scatter(
                *zip(*cali_image_coords),
                label="Calibration Points",
                s=60,
                c="mediumblue",
                alpha=0.8,
            )
            ax.scatter(cx, cy, label="Principal Point", s=70, c="crimson")

            plt.gca().update(
                {"title": image_path, "xlabel": "$u$ (px)", "ylabel": "$v$ (px)"}
            )
            plt.legend()

            if args.out is not None:
                out_path: str = args.out
                plt.savefig(out_path)
            else:
                plt.show()

    except AssertionError as e:
        print(f"\nERROR: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(f"\nKeyboard Interrupt")
        sys.exit(0)
