#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import calicam


def main():
    np.set_printoptions(precision=3, suppress=True)

    MAX_HELP_POSITION = 40

    parser = ArgumentParser(
        prog="calicam",
        description=(
            "Generates projection matrix and calculates intrinsic and extrinsic parameters.\n"
            "CSV inputs are in the format: x,y,z,u,v where 3D point = (x, y, z) and 2D point = (u,v)"),
        formatter_class=lambda prog: RawDescriptionHelpFormatter(prog, max_help_position=MAX_HELP_POSITION),
    )

    parser.add_argument("path", metavar="PATH", help="path to csv file with calibration points")
    parser.add_argument(
        "-d", "--data", metavar="DATA_PATH", help="path to csv file with model verification data")
    parser.add_argument(
        "-g", "--graph", nargs="?", const="", metavar="BKGD_IMG", help="generate graph")
    parser.add_argument("-t", "--title", metavar="TITLE", help="title of graph (ignored if `-g` is not passed)")
    parser.add_argument("-s", "--show", action="store_true", help="show graph (only necessary if `-o` is passed)")
    parser.add_argument("-o", "--out", metavar="GRAPH_PATH", help="graph output location ")
    parser.add_argument("--noprint", action="store_true", help="don't print output to terminal")

    args = parser.parse_args()

    output = []

    try:
        # GENERATE MODEL
        csv_path: str = args.path

        cali_world_coords, cali_image_coords = calicam.parse_data_from_csv(csv_path)
        proj_matrix, cali_matrix, rot_matrix, (tx, ty, tz) = calicam.calibrate_camera(
            cali_world_coords, cali_image_coords)

        # origin
        (ox, oy) = calicam.project_point(proj_matrix, (0.0, 0.0, 0.0))

        # principal point, focal lengths
        (cx, cy), (fx, fy) = calicam.extract_intrinsics(cali_matrix)
        
        # tait-bryan angles
        a, b, g = calicam.extract_orientation_zyx(rot_matrix)

        output.append("\n" + "\n\n".join((
            f"Projection Matrix: \n{proj_matrix}",
            f"Calibration Matrix: \n{cali_matrix}",
            f"Rotation Matrix: \n{rot_matrix}",
            f"Focal Lengths: \n\tf_x = {fx:.2f} px \n\tf_y = {fy:.2f} px",
            f"Principal Point: \n\tc_x = {cx:.2f} px \n\tc_y = {cy:.2f} px",
            f"Translation: \n\tt_x = {tx:.2f} \n\tt_y = {ty:.2f} \n\tt_z = {tz:.2f}",
            f"Orientation: \n\t\u03B1 = {a:.2f}° \n\t\u03B2 = {b:.2f}° \n\t\u03B3 = {g:.2f}°",
        )))

        # MODEL VALIDATION
        data_path: str | None = args.data

        if data_path is not None:
            assert os.path.isfile(data_path), f"{data_path} does not exist."
            assert data_path.endswith(".csv"), f'{data_path} does not end with the extension ".csv".'

            data_world_coords, data_image_coords = calicam.parse_data_from_csv(data_path)

            predicted_coords = [
                calicam.project_point(proj_matrix, world_coord) for world_coord in data_world_coords
            ]
            reproj_errs = [
                calicam.euclidean(actual_coord, reproj_coord)
                for actual_coord, reproj_coord in zip(data_image_coords, predicted_coords)
            ]

            max_err = max(reproj_errs)
            avg_err = sum(reproj_errs) / len(reproj_errs)

            output.append(
                f"\nReprojection Errors: \n\t\u03BC_max = {max_err:.3f} px \n\t\u03BC_avg = {avg_err:.3f} px")

        # OUTPUT
        if not args.noprint:
            print(*output, sep="\n")

        # GRAPH
        image_path: str | None = args.graph

        if image_path is not None:
            ax: plt.Axes
            _, ax = plt.subplots(figsize=(8, 10))

            plt.gca().invert_yaxis()

            # image was provided
            if image_path != "":
                assert os.path.isfile(image_path), f"{image_path} does not exist."
                assert args.data, "Path to data csv file must be provide using -d flag to produce graph."

                img = plt.imread(image_path,)
                ax.imshow(img, cmap='gray')
                ax.autoscale(False)

            # graph data points and model points only if -d flag is specfied
            if data_path is not None:
                cmap = plt.cm.brg
                discrete_cmap = list(cmap(np.linspace(0, 1, len(data_image_coords))))

                # data points
                ax.scatter(
                    *zip(*data_image_coords),
                    label="Ground Truth",
                    s=50,
                    color=discrete_cmap,
                    marker="o",
                    alpha=0.6,
                )

                # predicted points
                ax.scatter(
                    *zip(*predicted_coords),
                    label="Prediction",
                    s=100,
                    color=discrete_cmap,
                    marker="x",
                )

            # calibration points
            ax.scatter(
                *zip(*cali_image_coords),
                label="Calibration Points",
                s=60,
                marker="D",
                color="darkorange",
            )
            
            # origin point
            ax.scatter(ox, oy, label="Origin", s=120, marker="o", color="green")

            # principle point
            ax.scatter(cx, cy, label="Principal Point", s=120, marker="p", color="magenta")

            graph_title: str = args.title or image_path

            plt.gca().update({"title": graph_title, "xlabel": "$u$ (px)", "ylabel": "$v$ (px)"})
            plt.legend()

            out_path: str | None = args.out

            if out_path:
                plt.savefig(out_path, bbox_inches='tight')

            # show graph if -s flag was specified or if a save location was not specified
            if args.show or not out_path:
                plt.show()

    except AssertionError as e:
        parser.error(str(e))  # pass error to argparse

    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
