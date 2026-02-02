import FreeCAD as App
from FreeCAD import Units
import FreeCADGui, TechDrawGui
import Part
import random, time, os, math, argparse
from collections import deque

def create_cube(x, y, z, placement_space, doc, cube_size):
    """
    Create a cube at the specified (x, y, z) coordinates if the space is available.

    Args:
        x (int): The x-coordinate for the cube.
        y (int): The y-coordinate for the cube.
        z (int): The z-coordinate for the cube.
        placement_space (list): A 3D list representing the grid space (0 for empty, 1 for occupied).
        doc (FreeCAD.Document): The active FreeCAD document.
        cube_size (float): The size of the cube's side length.
    """
    cube = Part.makeBox(cube_size, cube_size, cube_size)
    obj = doc.addObject("Part::Feature", f"Cube_{x}_{y}_{z}")
    obj.Shape = cube
    if placement_space[z][y][x] == 0:
        obj.Placement.Base = App.Vector(x, y, z)
        placement_space[z][y][x] = 1
    # Update global bounding box dimensions
    global min_x, min_y, min_z, max_x, max_y, max_z
    min_x = min(min_x, x)
    min_y = min(min_y, y)
    min_z = min(min_z, z)
    max_x = max(max_x, x)
    max_y = max(max_y, y)
    max_z = max(max_z, z)

def connect_isolated_cubes(placement_space, space_size_x, space_size_y, doc, cube_size):
    """
    Ensures connectivity among cubes in the base layer (z=0) by finding and connecting
    any isolated regions.

    This function uses a breadth-first search (BFS) to identify separate clusters of cubes.
    If multiple clusters exist, it calculates the shortest path between the closest cubes
    of two clusters and fills in the path with new cubes to connect them.

    Args:
        placement_space (list): A 3D list representing the grid space.
        space_size_x (int): The size of the space along the x-axis.
        space_size_y (int): The size of the space along the y-axis.
        doc (FreeCAD.Document): The active FreeCAD document.
        cube_size (float): The side length of the cubes.
    """
    cubes = [(x, y) for x in range(space_size_x) for y in range(space_size_y) if placement_space[0][y][x] == 1]
    if not cubes:
        return  

    visited = set()
    regions = []
    
    # Find all connected regions on the base layer (z=0)
    for x, y in cubes:
        if (x, y) not in visited:
            region = []
            queue = deque()
            queue.append((x, y))
            visited.add((x, y))
            while queue:
                cx, cy = queue.popleft()
                region.append((cx, cy))
                # Check all 8 neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < space_size_x and 0 <= ny < space_size_y:
                        if (nx, ny) not in visited and placement_space[0][ny][nx] == 1:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            regions.append(region)
    
    # If more than one region is found, connect them
    if len(regions) > 1: 
        for i in range(len(regions) - 1):
            region1 = regions[i]
            region2 = regions[i + 1]
           
            # Find the closest pair of cubes between the two regions
            min_dist = float('inf')
            best_pair = None
            for x1, y1 in region1:
                for x2, y2 in region2:
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = ((x1, y1), (x2, y2))

            (x1, y1), (x2, y2) = best_pair
            # Walk from one cube to the other and create new cubes to fill the path
            dx = 1 if x2 > x1 else -1
            dy = 1 if y2 > y1 else -1
            x, y = x1, y1
            while x != x2 or y != y2:
                if x != x2 and y != y2:
                    dx = 1 if x2 > x else -1
                    dy = 1 if y2 > y else -1
                    x += dx
                    y += dy
                elif x != x2:
                    dx = 1 if x2 > x else -1
                    x += dx
                elif y != y2:
                    dy = 1 if y2 > y else -1
                    y += dy
                if placement_space[0][y][x] == 0:
                    create_cube(x, y, 0, placement_space, doc, cube_size)

def main(args):
    """
    Main function to generate 3D cube models and export various rotated and modified images.

    This function creates a random cube structure, ensures its connectivity, and filters
    it based on size. It then exports several isometric views, including a base view,
    rotations around the X, Y, and Z axes, and a final view with one cube randomly removed.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    space_size_x, space_size_y, space_size_z = args.space_size
    cube_size = 1

    generated = 0
    while generated < args.num_samples:
        placement_space = [[[0 for _ in range(space_size_x)] for _ in range(space_size_y)] for _ in range(space_size_z)]
        global min_x, min_y, min_z, max_x, max_y, max_z
        min_x = space_size_x
        min_y = space_size_y
        min_z = space_size_z
        max_x = 0
        max_y = 0
        max_z = 0

        doc = App.newDocument("Cubes")
        
        # Generate cubes in a random structure
        for z in range(space_size_z):
            for y in range(space_size_y):
                for x in range(space_size_x):
                    # Cubes can only be placed on the base layer (z=0) or on top of another cube
                    if z == 0 or (z > 0 and placement_space[z-1][y][x] == 1):
                        if random.choice([True, False]):
                            create_cube(x, y, z, placement_space, doc, cube_size)

        # Connect any isolated regions of cubes on the base layer
        connect_isolated_cubes(placement_space, space_size_x, space_size_y, doc, cube_size)
        
        # Filter generated models based on the number of solids
        cube_shapes = [obj.Shape for obj in doc.Objects if "Cube" in obj.Name]
        solids_count = len(cube_shapes)
        if solids_count < 8 or solids_count > 10:
            continue 
        
        # Calculate actual bounding box dimensions of the generated model
        actual_space_size_x = max_x - min_x + 1
        actual_space_size_y = max_y - min_y + 1
        actual_space_size_z = max_z - min_z + 1
        
        save_dir = f"{save_folder}/{i}-{actual_space_size_x}-{actual_space_size_y}-{actual_space_size_z}-{solids_count}"
        os.makedirs(save_dir, exist_ok=True) 

        # Save initial isometric view
        view = FreeCADGui.ActiveDocument.ActiveView
        view.viewIsometric()
        FreeCADGui.updateGui()
        view.fitAll()
        image_path = f"{save_dir}/original.png"
        view.saveImage(image_path, 1280, 1024)
        time.sleep(1)
            
        doc.recompute()

        # Create a single compound object for easier manipulation
        compound = Part.makeCompound(cube_shapes)
        compound_obj = doc.addObject("Part::Feature", "Compound")
        compound_obj.Shape = compound
        compound_obj.ViewObject.Visibility = False
        ori_placement_base = compound_obj.Placement.Base
    
        # Hide individual cube objects
        cube_objs = [obj for obj in doc.Objects if "Cube" in obj.Name]
        for cube in cube_objs:
            cube.ViewObject.Visibility = False
        
        axis_xyz = {(1, 0, 0):"x", (0, 1, 0):"y", (0, 0, 1):"z"}
        
        # Generate rotated and removed-cube views
        for r in range(5):
            rotation_axis = random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
            xyz = axis_xyz[rotation_axis]
            rotation_axis = App.Vector(rotation_axis) 
            rotation_angle = random.choice([90, 180, 270])
            if r == 4: # The last rotation is special
                rotation_angle = random.choice([0, 90, 180, 270])
            rotation = App.Rotation(rotation_axis, rotation_angle)
            
            if r < 4:
                # Create and place a new rotated compound object
                compound_obj_copy = doc.addObject("Part::Feature", "RotateCompound")
                compound_obj_copy.Shape = compound
                compound_obj_copy.Placement = App.Placement(ori_placement_base, rotation)
                image_path = f"{save_dir}/{xyz}-{rotation_angle}.png"
            else:
                # For the last view, remove one random cube
                cube_to_remove = random.choice(cube_objs)
                print(f"Removing cube: {cube_to_remove.Name}")

                remaining_shapes = [obj.Shape for obj in cube_objs if obj.Name != cube_to_remove.Name]
                remaining_compound = Part.makeCompound(remaining_shapes)
                remaining_compound_obj = doc.addObject("Part::Feature", "RemainCompound")
                remaining_compound_obj.Shape = remaining_compound
                remaining_compound_obj.Placement = App.Placement(ori_placement_base, rotation)
                image_path = f"{save_dir}/{xyz}-{rotation_angle}-remove.png"
            
            view = FreeCADGui.ActiveDocument.ActiveView
            view.viewIsometric()
            FreeCADGui.updateGui()
            view.fitAll()
            view.saveImage(image_path, 1280, 1024)
            time.sleep(1)
            compound_obj_copy.ViewObject.Visibility = False

        generated += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D cube rotation dataset")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to save generated data (e.g., /path/to/save/folder)")
    parser.add_argument("--space_size", type=int, default=[3, 3, 3], nargs=3, metavar=("X", "Y", "Z"),
                        help="Size of the space (default: 3x3x3)")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    args = parser.parse_args()
    main(args)


"""
    # run in FreeCAD
    import sys
    sys.argv = [
        "create_3D_rotation.py",
        "--save_folder", "/path/to/save",
        "--space_size", "3 3 3",
        "--num_samples", "50"
    ]
    exec(open("/path/to/create_3D_rotation.py").read())
"""