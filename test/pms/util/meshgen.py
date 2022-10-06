#!/bin/env python3
import pyvista as vista
import argparse

def main():
    parser = argparse.ArgumentParser(description="A tool to generate a cuboid VTK mesh")
    parser.add_argument("-o", metavar="FILE", type=str, help="Output file name", required=True)
    parser.add_argument("-Nx", type=int, default=None, help="Amount of grid cells along X axis", required=True)
    parser.add_argument("-Ny", type=int, default=None, help="Amount of grid cells along Y axis", required=True)
    parser.add_argument("-Nz", type=int, default=0,help="Amount of grid cells along Z axis", )
    parser.add_argument("-dx", type=float, default=None, help="Cell size along X axis", required=True)
    parser.add_argument("-dy", type=float, default=None, help="Cell size along Y axis", required=True)
    parser.add_argument("-dz", type=float, default=0, help="Cell size along Z axis", )
    args = parser.parse_args()

    # Generate a uniform grid
    output = vista.UniformGrid(dims=(args.Nx + 1, args.Ny + 1, args.Nz + 1), spacing=(args.dx, args.dy, args.dz))
    output = output.cast_to_unstructured_grid()
    # Tetrahedronize
    output = output.delaunay_3d(alpha = 0)
    # Center mesh along X and Y axes
    output = output.cell_data_to_point_data()
    output.points[:,0] -= args.Nx * args.dx / 2
    output.points[:,1] -= args.Ny * args.dy / 2
    # Save the mesh
    output.save(args.o)

if __name__ == "__main__":
    main()
