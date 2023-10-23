#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import calicam

def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to csv file")
    parser.add_argument("-g", "--graph", nargs=2, metavar=("data", "image"), type=str, help="Generate graph")
    
    np.set_printoptions(precision=4)

    args = parser.parse_args()
    
    calibration_world_coords, calibration_image_coords = calicam.parse_csv(args.path)
    _, proj_matrix = calicam.generate_proj_matrix(calibration_world_coords, calibration_image_coords)
    
    params = calicam.extract_parameters(*calicam.decompose_proj_matrix(proj_matrix))

    fx, fy = params.focal_lengths
    cx, cy = params.principal_point
    a, b, g = params.angles
    
    print('\n'.join((
        f"Projection Matrix:\n {proj_matrix}",
        f"Focal Lengths: \n\tfx = {fx:.2f} px\n\tfy = {fy:.2f} px",
        f"Principal Point: \n\tcx = {cx:.2f} px\n\tcy = {cy:.2f} px",
        f"Orientation: \n\tα = {a:.2f}° \n\tβ = {b:.2f}° \n\tγ = {g:.2f}°",
    )))

    if args.graph:
        data_path, image_path = args.graph
    
        data_world_coords, data_image_coords = calicam.parse_csv(data_path)
        reprojections = calicam.project(proj_matrix, data_world_coords)
        _, avg_error = calicam.calculate_reproj_error(data_image_coords, reprojections)
        
        print(f"\nAverage Reprojection Error: \n\tμ = {avg_error:.2f} px")
        
        _ , ax = plt.subplots(figsize=(8, 10))
        
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.autoscale(False)
        
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        
        ax.plot(((x_min + x_max) / 2, (x_min + x_max) / 2), (y_min, y_max), ls='--', lw=0.4, c="gray")
        ax.plot((x_min, x_max), ((y_min + y_max) / 2, (y_min + y_max) / 2), ls='--', lw=0.4, c="gray")
            
        ax.scatter(cx, cy, label='Principal Point', s=60, c='crimson')
        ax.scatter(*zip(*calibration_image_coords), label='Calibration Points', s=60, c="blue", alpha=0.8)
        ax.scatter(*zip(*reprojections), label='Reprojected Points', s=30, c="yellow", alpha=0.8)
        
        graph_info = '\n'.join((
            rf"$f_x$ = ${fx:.2f}$ px, $f_y$ = ${fy:.2f}$ px",
            rf"$c_x$ = ${cx:.2f}$ px, $c_y$ = ${cy:.2f}$ px",
            rf"$\alpha$ = ${a:.2f}°$, $\beta$ = ${b:.2f}°$, $\gamma$ = ${g:.2f}$°",
            rf"$\mu$ = $\pm {avg_error:.2f}$ px"
        ))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.03, 0.03, graph_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

        plt.gca().update({
            "title": image_path, 
            "xlabel":'u (px)', 
            "ylabel":'v (px)'})
        plt.legend()
        plt.show()
        
    return 0
    
if __name__ == "__main__":
    main()
