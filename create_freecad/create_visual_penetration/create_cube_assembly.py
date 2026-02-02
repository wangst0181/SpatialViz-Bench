import FreeCAD as App
from FreeCAD import Units
import FreeCADGui, TechDrawGui
import Part
import random, time, os
from collections import deque
import argparse

def create_cube(x, y, z, placement_space):
    """
    Creates a single cube at the specified coordinates and adds it to the global cube dictionary.

    Args:
        x (int): The x-coordinate for the cube's position.
        y (int): The y-coordinate for the cube's position.
        z (int): The z-coordinate for the cube's position.
        placement_space (list): A 3D list representing the grid to track cube placement.
    """
    global cubes
    if placement_space[z][y][x] == 0:
        cube = Part.makeBox(cube_size, cube_size, cube_size)
        obj = doc.addObject("Part::Feature", f"Cube_{x}_{y}_{z}")
        obj.Shape = cube
        obj.Placement.Base = App.Vector(x, y, z)
        placement_space[z][y][x] = 1
        cubes[(x,y,z)] = obj

def get_neighbors(cube_pos, cubes):
    """
    Finds and returns the neighboring cube positions for a given cube.

    Args:
        cube_pos (tuple): The (x, y, z) position of the cube.
        cubes (dict): A dictionary mapping cube positions to their FreeCAD objects.

    Returns:
        list: A list of neighboring cube positions.
    """
    x, y, z = cube_pos
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if abs(dx) + abs(dy) + abs(dz) == 1:  
                    neighbor_pos = (x + dx, y + dy, z + dz)
                    if neighbor_pos in cubes:
                        neighbors.append(neighbor_pos)
    return neighbors

def region_growing(cubes, max_cubes):
    """
    Performs a region-growing algorithm to select a continuous part of the cube structure.

    Args:
        cubes (dict): A dictionary mapping cube positions to their FreeCAD objects.
        max_cubes (int): The maximum number of cubes allowed in the growing region.

    Returns:
        set: A set of positions belonging to the grown region.
    """
    if not cubes:
        return set()
    start_pos = random.choice(list(cubes.keys()))
    part = set()
    queue = deque([start_pos])
    
    while queue and len(part) < max_cubes:
        current_pos = queue.popleft()
        if current_pos not in part:
            part.add(current_pos)
            neighbors = get_neighbors(current_pos, cubes)
            queue.extend([n for n in neighbors if n not in part])
    
    return part

def is_continuous(part, cubes):
    """
    Checks if a given set of cubes forms a single, continuous, connected component.

    Args:
        part (set): A set of cube positions to check.
        cubes (dict): A dictionary mapping cube positions to their FreeCAD objects.

    Returns:
        bool: True if the part is continuous, False otherwise.
    """
    if not part:
        return False
    start_pos = next(iter(part))
    visited = set()
    queue = deque([start_pos])
    
    while queue:
        current_pos = queue.popleft()
        if current_pos not in visited:
            visited.add(current_pos)
            neighbors = get_neighbors(current_pos, cubes)
            queue.extend([n for n in neighbors if n in part and n not in visited])
    
    return len(visited) == len(part)

def split_cubes(cubes, max_percentage, num_parts=2):
    """
    Splits the main cube structure into a specified number of continuous parts.

    Args:
        cubes (dict): The dictionary of all cubes in the structure.
        max_percentage (float): The maximum allowed size of a part relative to the total number of cubes.
        num_parts (int): The number of parts to split the structure into (2 or 3).

    Returns:
        tuple: A tuple containing the sorted parts (by size), or a tuple of None if splitting fails.
    """
    total_cubes = len(cubes)
    max_cubes_per_part = int(total_cubes * max_percentage)
    original_cubes = dict(cubes)

    part1_positions = None
    remaining_after_part1 = None
    cnt = 0
    while cnt < 5:
        candidate = region_growing(original_cubes, max_cubes_per_part)
        if not is_continuous(candidate, original_cubes):
            cnt += 1
            continue
        remaining = set(original_cubes.keys()) - candidate
        if is_continuous(remaining, original_cubes):
            part1_positions = candidate
            remaining_after_part1 = remaining
            break
        cnt += 1

    if part1_positions is None:
        return (None, None) if num_parts == 2 else (None, None, None)

    part1 = {pos: original_cubes[pos] for pos in part1_positions}
    
    if num_parts == 2:
        part2 = {pos: original_cubes[pos] for pos in remaining_after_part1}
        cubes.clear()
        return sorted([part1, part2], key=len)
    
    part2_positions = None
    remaining_after_part2 = None
    cubes_after_part1 = {k: original_cubes[k] for k in remaining_after_part1}
    
    cnt = 0
    while cnt < 5:
        candidate = region_growing(cubes_after_part1, max_cubes_per_part)
        if not is_continuous(candidate, cubes_after_part1):
            cnt += 1
            continue
        remaining = set(cubes_after_part1.keys()) - candidate
        if is_continuous(remaining, cubes_after_part1):
            part2_positions = candidate
            remaining_after_part2 = remaining
            break
        cnt += 1

    if part2_positions is None:
        return None, None, None
    
    part2 = {pos: original_cubes[pos] for pos in part2_positions}
    part3 = {pos: original_cubes[pos] for pos in remaining_after_part2}

    cubes.clear()
    return sorted([part1, part2, part3], key=len)

def main(args):
    """
    The main function to generate sliding block puzzles and their corresponding images.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    num_samples = args.num_samples
    save_folder = args.save_folder
    num_parts = args.num_parts
    space_size_x, space_size_y, space_size_z = args.space_size
    global cube_size
    cube_size = 1
    
    for i in range(num_samples):
        global cubes
        placement_space = [[[0 for _ in range(space_size_x)] for _ in range(space_size_y)] for _ in range(space_size_z)]
        cubes = {}

        doc = App.newDocument("Cubes")
        
        k=1
        for y in range(space_size_y):
            k = random.randint(k, min(y+2, space_size_x))
            for x in range(k):
                create_cube(x, y, 0, placement_space)
        
        for z in range(1, space_size_z-1):
            k=0
            for y in range(space_size_y):
                k = random.randint(k, max(k, sum(placement_space[z-1][y])))
                for x in range(k):
                    create_cube(x, y, z, placement_space)
        
        z = space_size_z-1
        for y in range(space_size_y):
            for x in range(space_size_x):
                if placement_space[z-1][y][x]==1:
                    if random.choice([True, False]):
                        create_cube(x, y, z, placement_space)      
        
        solids_count = len([obj for obj in doc.Objects if "Cube" in obj.Name])
        if solids_count < 8:
            App.closeDocument(doc.Name)
            continue      
        
        if num_parts == 2:
            part1, part2 = split_cubes(cubes, 0.6, num_parts=2)
            if part1 is None or part2 is None:
                App.closeDocument(doc.Name)
                continue
            assert (len(part1) + len(part2)) == solids_count
        else:
            part1, part2, part3 = split_cubes(cubes, 0.4, num_parts=3)
            if part1 is None or part2 is None or part3 is None:
                App.closeDocument(doc.Name)
                continue
            assert (len(part1) + len(part2) + len(part3)) == solids_count
        if part1 is None:
            App.closeDocument(doc.Name)
            continue
        
        save_dir = os.path.join(save_folder, f"{i}-{space_size_x}-{space_size_y}-{space_size_z}-{solids_count}-{num_parts}")
        os.makedirs(save_dir, exist_ok=True) 

        view = FreeCADGui.ActiveDocument.ActiveView
        view.viewIsometric()
        FreeCADGui.updateGui()
        time.sleep(1)
        view.fitAll()
        
        image_path = os.path.join(save_dir, "total.png")
        view.saveImage(image_path, 1280, 1024)
        time.sleep(1)
        
        part_obj1 = []
        part_obj2 = []
        for pos, cube in part1.items():
            part_obj1.append(cube.Shape)
        for pos, cube in part2.items():
            part_obj2.append(cube.Shape)

        if num_parts == 3:
            part_obj3 = []
            for pos, cube in part3.items():
                part_obj3.append(cube.Shape)
        
        if num_parts == 2:
            part_obj_incorrect = part_obj2
        elif num_parts == 3:
            part_obj_incorrect = part_obj3
            
        target_ids = random.sample(list(range(0, len(part_obj_incorrect))), 2)
        incorrect_part_obj_1 = list(filter(lambda obj: id(obj) != id(part_obj_incorrect[target_ids[0]]), part_obj_incorrect))
        incorrect_part_obj_2 = list(filter(lambda obj: id(obj) != id(part_obj_incorrect[target_ids[1]]), part_obj_incorrect))

        compound1 = Part.makeCompound(part_obj1)
        compound2 = Part.makeCompound(part_obj2)
        incorrect_compound_1 = Part.makeCompound(incorrect_part_obj_1)
        incorrect_compound_2 = Part.makeCompound(incorrect_part_obj_2)
        
        compound_obj1 = doc.addObject("Part::Feature", "Cube_Compound1")
        compound_obj2 = doc.addObject("Part::Feature", "Cube_Compound2")
        incorrect_compound_obj_1 = doc.addObject("Part::Feature", "Cube_Incorrect_Compound_1")
        incorrect_compound_obj_2 = doc.addObject("Part::Feature", "Cube_Incorrect_Compound_2")
        
        compound_obj1.Shape = compound1
        compound_obj2.Shape = compound2
        incorrect_compound_obj_1.Shape = incorrect_compound_1
        incorrect_compound_obj_2.Shape = incorrect_compound_2
        
        compounds = [compound_obj1, compound_obj2, incorrect_compound_obj_1, incorrect_compound_obj_2]
        
        if num_parts == 3:
            compound3 = Part.makeCompound(part_obj3)
            compound_obj3 = doc.addObject("Part::Feature", "Cube_Compound3")
            compound_obj3.Shape = compound3
        
            compounds = [compound_obj1, compound_obj2, compound_obj3, incorrect_compound_obj_1, incorrect_compound_obj_2]
        
        for cube_obj in [obj for obj in doc.Objects if "Cube" in obj.Name]:
            cube_obj.ViewObject.Visibility = False
        
        for j, compound_obj in enumerate(compounds):
            compound_obj.ViewObject.Visibility = True
            
            view = FreeCADGui.ActiveDocument.ActiveView
            view.viewIsometric()
            FreeCADGui.updateGui()
            time.sleep(1)
            
            view.fitAll()
            
            if "Incorrect" in compound_obj.Name:
                image_path = os.path.join(save_dir, f"incorrect_compound_{num_parts-1}_{j-num_parts}.png")
            else:
                image_path = os.path.join(save_dir, f"compound_{j}.png")
            view.saveImage(image_path, 1280, 1024)
            time.sleep(1)
            
            compound_obj.ViewObject.Visibility = False
        
        App.closeDocument(doc.Name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sliding block puzzles and associated images.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to the directory to save the generated images.")
    parser.add_argument("--space_size", type=int, default=[3, 3, 3], nargs=3, metavar=("X", "Y", "Z"),
                        help="Size of the space (default: 3x3x3)")
    parser.add_argument("--num_parts", type=int, default=2, help="Number of parts to assemble.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of data samples to generate.")
    args = parser.parse_args()
    main(args)
    
"""
    # run in FreeCAD
    import sys

    sys.argv = [
        "create_cube_assembly.py",
        "--save_folder", "/path/to/save",
        "--space_size", "3 3 3",
        "--num_parts", "2",
        "--num_samples", "50"
    ]

    exec(open("/path/to/create_cube_assembly.py").read())

"""