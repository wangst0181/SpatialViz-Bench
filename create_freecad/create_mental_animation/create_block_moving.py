import FreeCAD as App
from FreeCAD import Units
import FreeCADGui, TechDrawGui
import Part, Draft
from pivy import coin
import random, time, os, json
from collections import deque
from copy import deepcopy
import numpy as np
import argparse

def create_cube(x, y, z, placement_space, cubes, cubes_color, color):
    """
    Creates a single colored cube at the specified coordinates and adds it to the global cube dictionaries.
    
    Args:
        x (int): The x-coordinate for the cube's position.
        y (int): The y-coordinate for the cube's position.
        z (int): The z-coordinate for the cube's position.
        placement_space (list): A 3D list representing the grid to track cube placement.
        cubes (dict): A dictionary mapping cube positions to their FreeCAD objects.
        cubes_color (dict): A dictionary mapping cube positions to their colors.
        color (list): The RGB color value of the cube.
    """
    cube = Part.makeBox(cube_size, cube_size, cube_size)
    obj = doc.addObject("Part::Feature", f"Cube_{x}_{y}_{z}")
    obj.Shape = cube
    obj.ViewObject.DiffuseColor = [color] * 6
    obj.ViewObject.DisplayMode = "Flat Lines"
    obj.ViewObject.Transparency = 0 
    obj.ViewObject.update()
    if placement_space[z][y][x] == 0:
        obj.Placement.Base = App.Vector(x, y, z)
        placement_space[z][y][x] = 1
        cubes[(x,y,z)] = obj
        cubes_color[(x,y,z)] = color

def is_valid_position(pos):
    """
    Checks if a given position is within the grid boundaries.
    
    Args:
        pos (tuple): The (x, y, z) position to check.
    
    Returns:
        bool: True if the position is valid, False otherwise.
    """
    x, y, z = pos
    return (0 <= x < space_size_x and 
            0 <= y < space_size_y and 
            0 <= z < space_size_z
        )

def has_support(pos, cubes):
    """
    Checks if a cube at a given position is supported from below.
    
    Args:
        pos (tuple): The (x, y, z) position of the cube.
        cubes (dict): The dictionary of all active cubes.
    
    Returns:
        bool: True if the cube has support, False otherwise.
    """
    x, y, z = pos
    if z == 0:  
        return True
    if (x, y, z - 1) in cubes:  
        return True
    return False

def drop_cubes(cubes, cubes_color, simulate=True):
    """
    Simulates gravity by dropping unsupported cubes.
    
    Args:
        cubes (dict): The dictionary of FreeCAD cube objects.
        cubes_color (dict): The dictionary mapping cube positions to colors.
        simulate (bool): If True, only simulate the color change; otherwise, move the actual FreeCAD objects.
    """
    moved = True
    while moved:
        moved = False
        for pos in list(cubes_color.keys()):
            x, y, z = pos
            if z > 0 and (x, y, z - 1) not in cubes_color:
                new_pos = (x, y, z - 1)
                
                if not simulate:
                    cube = cubes[pos]
                    cube.Placement.Base = App.Vector(x, y, z - 1)
                    cubes[new_pos] = cube
                    del cubes[pos]
                
                cubes_color[new_pos] = cubes_color[pos]
                del cubes_color[pos]
                moved = True
        if not simulate:
            doc.recompute()
            FreeCADGui.updateGui()

def move_cube(from_pos, to_pos, cubes, cubes_color, simulate=True):
    """
    Moves a cube from a starting position to a target position. Handles swaps if the target is occupied.
    
    Args:
        from_pos (tuple): The starting (x, y, z) position.
        to_pos (tuple): The target (x, y, z) position.
        cubes (dict): The dictionary of FreeCAD cube objects.
        cubes_color (dict): The dictionary mapping cube positions to colors.
        simulate (bool): If True, only simulate the color change; otherwise, move the actual FreeCAD objects.
        
    Returns:
        bool: True if the move was successful, False otherwise.
    """
    if not is_valid_position(to_pos) or from_pos not in cubes_color:
        return False
    
    if (to_pos not in cubes_color and not has_support(to_pos, cubes_color)):
        return False

    
    if to_pos not in cubes_color:
        if not simulate:
            cube = cubes[from_pos]
            cube.Placement.Base = App.Vector(to_pos[0], to_pos[1], to_pos[2])
            cubes[to_pos] = cube
            del cubes[from_pos]
        
        cubes_color[to_pos] = cubes_color[from_pos]
        del cubes_color[from_pos]
        drop_cubes(cubes, cubes_color, simulate=simulate)
        return True
    
    elif to_pos in cubes_color:
        if not simulate:
            cube1 = cubes[from_pos]
            cube2 = cubes[to_pos]
            cube1.Placement.Base = App.Vector(to_pos[0], to_pos[1], to_pos[2])
            cube2.Placement.Base = App.Vector(from_pos[0], from_pos[1], from_pos[2])
            cubes[to_pos] = cube1
            cubes[from_pos] = cube2
 
        cubes_color[to_pos], cubes_color[from_pos] = (cubes_color[from_pos], cubes_color[to_pos])
        drop_cubes(cubes, cubes_color, simulate=simulate)
        return True
    return False

def generate_random_transformation(cubes, cubes_color, num_moves=3):
    """
    Generates a sequence of random, valid moves for the cubes.
    
    Args:
        cubes (dict): The dictionary of FreeCAD cube objects.
        cubes_color (dict): The dictionary mapping cube positions to colors.
        num_moves (int): The number of moves to generate.
    
    Returns:
        list: A list of tuples, where each tuple represents a move: ((from_pos), (direction)).
    """
    transformations = []
    
    for _ in range(num_moves):
        possible_moves = []
        for from_pos in cubes:
            for direction in directions:
                to_pos = tuple(a + b for a, b in zip(from_pos, direction))
                if is_valid_position(to_pos) and (to_pos in cubes or (to_pos not in cubes and has_support(to_pos, cubes))):
                    possible_moves.append((from_pos, direction))
        
        if not possible_moves:
            break
            
        from_pos, direction = random.choice(possible_moves)
        to_pos = tuple(a + b for a, b in zip(from_pos, direction))
        if move_cube(from_pos, to_pos, cubes, cubes_color, simulate=False):
            transformations.append((from_pos, direction))
    doc.recompute()
    return transformations

def check_if_diff(cubes_color1, cubes_color2):
    """
    Compares two cube color configurations to check if they are different.
    
    Args:
        cubes_color1 (dict): The first cube color configuration.
        cubes_color2 (dict): The second cube color configuration.
    
    Returns:
        bool: True if the configurations are different, False otherwise.
    """
    if set(cubes_color1.keys()) != set(cubes_color2.keys()):
        return True
        
    for pos in cubes_color1:
        if cubes_color1[pos] != cubes_color2[pos]:
            return True
    return False
            

def generate_incorrect_random_transformation(cubes, cubes_color, cubes_color_correct, num_moves=3):
    """
    Generates an incorrect sequence of moves that leads to a different final state.
    
    Args:
        cubes (dict): The dictionary of FreeCAD cube objects.
        cubes_color (dict): The dictionary mapping cube positions to colors for the simulation.
        cubes_color_correct (dict): The correct final color configuration to compare against.
        num_moves (int): The number of moves to generate.
    
    Returns:
        list: A list of tuples representing the incorrect transformation. Returns an empty list if no valid incorrect transformation can be found.
    """
    cubes_color_copy = deepcopy(cubes_color)
    cnt = 0
    while True:
        transformations = []
        # Reinitialize for each attempt
        temp_cubes_color = deepcopy(cubes_color)
        
        for _ in range(num_moves):
            possible_moves = []
            for from_pos in temp_cubes_color:
                for direction in directions:
                    to_pos = tuple(a + b for a, b in zip(from_pos, direction))
                    if is_valid_position(to_pos) and (to_pos in temp_cubes_color or (to_pos not in temp_cubes_color and has_support(to_pos, temp_cubes_color))):
                        possible_moves.append((from_pos, direction))
            
            if not possible_moves:
                break
                
            from_pos, direction = random.choice(possible_moves)
            to_pos = tuple(a + b for a, b in zip(from_pos, direction))
            if move_cube(from_pos, to_pos, cubes, temp_cubes_color, simulate=True):
                transformations.append((from_pos, direction))
        
        diff = check_if_diff(cubes_color_correct, temp_cubes_color)
        print("incorrect_diff", diff)
        
        if diff or cnt > 5:
            if not diff:
                return []
            break   
        cnt += 1
    
    return transformations

def draw_axis(origin, direction, color, label):
    """
    Draws a single axis line with a label using the Draft workbench.
    
    Args:
        origin (App.Vector): The starting point of the axis.
        direction (tuple): The (x, y, z) vector representing the axis direction.
        color (tuple): The RGB color value of the axis.
        label (str): The text label for the axis.
    """
    end_point = origin + App.Vector(direction)
    line = Draft.make_line(origin, end_point)
    line.ViewObject.ArrowType = "Arrow" 
    line.ViewObject.ArrowSize = 0.1
    line.ViewObject.LineWidth = 3.0 
    line.ViewObject.LineColor = color
    line.ViewObject.EndArrow=True 
    label_pos = end_point 
    text = Draft.make_text(label, label_pos)
    text.ViewObject.TextColor = color
    text.ViewObject.FontSize = 0.5
    doc.recompute()

def main(args):
    """
    Main function to generate blocks moving puzzles and their corresponding data.
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    num_samples = args.num_samples
    save_folder = args.save_folder
    
    global cube_size, space_size_x, space_size_y, space_size_z, num_moves, directions, doc, cubes_color
    
    cube_size = 1
    space_size_x, space_size_y, space_size_z = args.space_size
    num_moves = args.num_moves
    
    color_value = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "pink": (1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0)
    }
    colors = list(color_value.values())
    directions = [
        (1, 0, 0), (-1, 0, 0),  
        (0, 1, 0), (0, -1, 0),  
        (0, 0, 1), (0, 0, -1)   
    ] 

    i = 0
    while i < num_samples:
        save_dir = os.path.join(save_folder, f"{i}-{space_size_x}-{space_size_y}-{space_size_z}-{num_moves}")
        os.makedirs(save_dir, exist_ok=True)
        
        doc = App.newDocument("Cubes")

        placement_space = [[[0 for _ in range(space_size_x)] for _ in range(space_size_y)] for _ in range(space_size_z)]
        cubes = {}
        cubes_color = {}

        for y in range(space_size_y):
            for x in range(space_size_x):
                square = Part.makePlane(cube_size, cube_size, App.Vector(x, y, 0))
                obj = doc.addObject("Part::Feature", f"Square_{x}_{y}")
                obj.Shape = square

        num = int(space_size_x * space_size_y * space_size_z * 0.5)
        choices = [True, False]
        for z in range(space_size_z):
            for y in range(space_size_y):
                for x in range(space_size_x):
                    if z == 0 or (z>0 and placement_space[z-1][y][x] == 1):
                        if random.choice(choices):
                            create_cube(x, y, z, placement_space, cubes, cubes_color, color=random.choice(colors))
                        if np.sum(np.array(placement_space)==1)>num:
                            choices = [False]
        
        origin = App.Vector(0, 0, 0)
        draw_axis(origin, (space_size_x+2, 0, 0), (1.0, 0.0, 0.0), "X") 
        draw_axis(origin, (0, space_size_y+2, 0), (0.0, 1.0, 0.0), "Y")  
        draw_axis(origin, (0, 0, space_size_z+1), (0.0, 0.0, 1.0), "Z")  
                               
        if len(cubes) < 3:
            App.closeDocument(doc.Name)
            continue

        doc.recompute()
                            
        view = FreeCADGui.ActiveDocument.ActiveView
        view.viewAxonometric()
        FreeCADGui.updateGui()
        view.fitAll()
        time.sleep(1)
        image_path = os.path.join(save_dir, "init.png")
        view.saveImage(image_path, 1280, 1024)
        time.sleep(1)

        view = FreeCADGui.ActiveDocument.ActiveView
        cam = view.getCameraNode()
        rot = coin.SbRotation(coin.SbVec3f(0, 0, 1), 3.14159) 
        current_orientation = cam.orientation.getValue()
        new_orientation = current_orientation * rot
        cam.orientation.setValue(new_orientation)
        FreeCADGui.updateGui()
        view.fitAll()
        time.sleep(1)
        image_path = os.path.join(save_dir, "init_back.png")
        view.saveImage(image_path, 1280, 1024)
        view.fitAll()

        cubes_color_init = deepcopy(cubes_color)
        
        json_data = {}                        
        transformations = generate_random_transformation(cubes, cubes_color, num_moves=3)
        json_data['correct'] = transformations
        
        doc.recompute()

        view = FreeCADGui.ActiveDocument.ActiveView
        view.viewAxonometric()
        FreeCADGui.updateGui()
        view.fitAll()
        time.sleep(1)
        image_path = os.path.join(save_dir, "transformed.png")
        view.saveImage(image_path, 1280, 1024)
        time.sleep(1)

        view = FreeCADGui.ActiveDocument.ActiveView
        cam = view.getCameraNode()
        rot = coin.SbRotation(coin.SbVec3f(0, 0, 1), 3.14159) 
        current_orientation = cam.orientation.getValue()
        new_orientation = current_orientation * rot
        cam.orientation.setValue(new_orientation)
        FreeCADGui.updateGui()
        view.fitAll()
        time.sleep(1)
        image_path = os.path.join(save_dir, "transformed_back.png")
        view.saveImage(image_path, 1280, 1024)
        time.sleep(1)

        diff = check_if_diff(cubes_color_init, cubes_color) 

        j = 0
        json_data['incorrect'] = []
        while j < 10:
            j += 1
            incorrect_transformations = generate_incorrect_random_transformation(cubes, cubes_color_init, cubes_color, num_moves=3)
            if not incorrect_transformations:
                continue
            json_data['incorrect'].append(incorrect_transformations)
            if len(json_data['incorrect']) == 3:
                break
            
        if len(json_data['incorrect'])<3:
            App.closeDocument(doc.Name)
            continue
        
        with open(os.path.join(save_dir, 'info.json'), 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False)
        
        App.closeDocument(doc.Name)
        i += 1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 3D blocks moving puzzles.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to the directory to save the generated images and JSON files.")
    parser.add_argument("--space_size", type=int, default=[3, 3, 3], nargs=3, metavar=("X", "Y", "Z"),
                        help="Size of the space (default: 3x3x3)")
    parser.add_argument("--num_moves", type=int, default=3, help="The number of moves of the transformation.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of data samples to generate.")

    args = parser.parse_args()
    main(args)
    
"""
    # run in FreeCAD
    import sys
    sys.argv = [
        "create_block_moving.py",
        "--save_folder", "/path/to/save",
        "--space_size", "3 3 3",
        "--num_moves", "3",
        "--num_samples", "50"
    ]
    exec(open("/path/to/create_block_moving.py").read())
"""