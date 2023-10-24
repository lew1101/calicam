#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import sys

import calicam

def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("path", help="Path to csv file")
    parser.add_argument("-d", "--data", help="Path to csv file")
    parser.add_argument("-g", "--graph", nargs='?', default=None, const="", help="Generate graph")
    
    np.set_printoptions(precision=4)

    args = parser.parse_args()
    
    calibration_world_coords, calibration_image_coords = calicam.parse_csv(args.path)
    _, proj_matrix = calicam.generate_proj_matrix(calibration_world_coords, calibration_image_coords)
    
    params = calicam.extract_parameters(proj_matrix)

    fx, fy = params.focal_lengths
    cx, cy = params.principal_point
    a, b, g = params.angles
    tx, ty, tz = params.translation_vector
    
    print('\n\n'.join((
        f"\nProjection Matrix:\n {proj_matrix}\n",
        f"Focal Lengths: \n\tfx = {fx:.2f} px\n\tfy = {fy:.2f} px",
        f"Principal Point: \n\tcx = {cx:.2f} px\n\tcy = {cy:.2f} px",
        f"Orientation: \n\t\u03b1 = {a:.2f}° \n\t\u03b2 = {b:.2f}° \n\t\u03b3 = {g:.2f}°",
        f"Translation: \n\tt = [{tx:.2f}, {ty:.2f}, {tz:.2f}] mm"
    )))
    
    if args.data:
        data_path = args.data
        
        data_world_coords, data_image_coords = calicam.parse_csv(data_path)
        reprojections = calicam.project(proj_matrix, data_world_coords)
        _, avg_error = calicam.calculate_reproj_error(data_image_coords, reprojections)
        
        print(f"\nAverage Reprojection Error: \n\tμ = {avg_error:.3f} px")

    if args.graph != None:
        image_path = args.graph
        
        if not args.data:
            raise Exception("Path to data csv file must be provided using -d flag")
        
        _ , ax = plt.subplots(figsize=(8, 10))
        
        plt.gca().invert_yaxis()
        
        if image_path != "":
            img = plt.imread(image_path)
            ax.imshow(img)
            ax.autoscale(False)
            
        ax.scatter(cx, cy, label='Principal Point', s=60, c='crimson')
        ax.scatter(*zip(*reprojections), label='Reprojected Points', s=40, c="yellow", alpha=0.8)
        ax.scatter(*zip(*calibration_image_coords), label='Calibration Points', s=60, c="blue", alpha=0.8)
        
        graph_info = '\n'.join((
            rf"$f_x$ = ${fx:.2f}$ px, $f_y$ = ${fy:.2f}$ px",
            rf"$c_x$ = ${cx:.2f}$ px, $c_y$ = ${cy:.2f}$ px",
            rf"$\alpha$ = ${a:.2f}°$, $\beta$ = ${b:.2f}°$, $\gamma$ = ${g:.2f}$°",
            rf"$\mu$ = $\pm {avg_error:.2f}$ px",
            rf"$\vec{{t}}$ = $[{tx:.2f}, {ty:.2f}, {tz:.2f}]$ mm"
        ))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.03, 0.03, graph_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

        plt.gca().update({
            "title": image_path, 
            "xlabel":'$u$ (px)', 
            "ylabel":'$v$ (px)'})
        plt.legend()
        plt.show()
        
    return 0
    
if __name__ == "__main__":
    main()
