from collections import deque
import Part
import FreeCADGui
import random, json, os, time
import sys
import argparse

def create_squares(matrix, faces, colors, save_folder, save_name=None):
    """
    Creates a 2D net of squares based on a matrix and face assignments.
    
    This function generates a 2D representation of a cube net using FreeCAD's
    Part module. It creates a square for each face defined in the 'faces'
    dictionary and colors it according to the 'colors' dictionary. The net is
    then rendered and an image is saved.

    Args:
        matrix (list of list of int): A 2D list representing the cube net.
        faces (dict): A dictionary mapping face names (e.g., "Top", "Front")
                      to their (row, column) coordinates in the matrix.
        colors (dict): A dictionary mapping face names to their RGB color tuples.
        save_folder (str): The path to the directory where the output will be saved.
        save_name (str, optional): The name for the sample, used for the
                                   output folder and file names. Defaults to None.
    """
    doc = App.newDocument("MatrixSquares")
    rows, cols = len(matrix), len(matrix[0])
    
    unit_size=1
    for face_name in faces:
        i, j = faces[face_name]
        x = j * unit_size
        y = (rows - 1 - i) * unit_size  
        square = Part.makePlane(unit_size, unit_size, App.Vector(x, y, 0))
        obj = doc.addObject("Part::Feature", f"Square_{i}_{j}")
        obj.Shape = square
        
        color = colors[face_name]
        obj.ViewObject.ShapeColor = color
        obj.ViewObject.DiffuseColor = [color]  
        obj.ViewObject.LineColor = (0.0, 0.0, 0.0)
        obj.ViewObject.DisplayMode = "Flat Lines"
        obj.ViewObject.Transparency = 0 
        obj.ViewObject.update()
    
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewIsometric()
    FreeCADGui.updateGui()
    view.fitAll()
    image_path = f"{save_folder}/{save_name}/{save_name}_unfold.png"
    view.saveImage(image_path, 1280, 1024)

def identify_cube_faces(matrix, base_face):
    """
    Identifies the faces of a cube net from a 2D matrix.

    This function uses a breadth-first search (BFS) algorithm to traverse the
    2D matrix and identify the positions of each of the six cube faces
    (Top, Bottom, Front, Back, Left, Right) relative to a base face.

    Args:
        matrix (list of list of int): A 2D list representing the cube net,
                                      where 1s are parts of the net and 0s are empty space.
        base_face (tuple): A tuple containing the name of the base face and
                           its (row, column) coordinates, e.g., ("Bottom", 1, 1).

    Returns:
        dict: A dictionary mapping face names to their (row, column)
              coordinates in the matrix.
    """
    faces = {}  
    visited = {}  
    
    face_name, r, c = base_face
    faces[face_name] = (r, c)
    
    # Define movement rules for unfolding the cube
    adjacent_moves = {
        "Bottom": [("Right", (0, 1)), ("Front", (1, 0)), ("Back", (-1, 0))],
        "Right": [("Top", (0, 1)), ("Front", (1, 0)), ("Back", (-1, 0))],
        "Front": [("Top", (0, 1))],
        "Back": [("Left", (0, -1))],
        "Top": [("Left", (0, 1)), ("Front", (1, 0))],
        "Left": [("Front", (0, -1)), ("Front", (1, 0))],
    }

    queue = deque([(face_name, r, c)])
    visited[(r, c)] = face_name
    while queue:
        current_face, r, c = queue.popleft()
        for new_face, (dr, dc) in adjacent_moves[current_face]:
            nr, nc = r + dr, c + dc
            if new_face not in faces:
                if 0 <= nr < len(matrix) and 0 <= nc < len(matrix[0]) and matrix[nr][nc] == 1 and (nr, nc) not in visited:
                    faces[new_face] = (nr, nc)
                    queue.append((new_face, nr, nc))
                    visited[(nr, nc)] = new_face
    return faces

def make_colored_cube(colors, save_folder, save_name):
    """
    Creates a 3D colored cube and saves images from different viewpoints.

    This function generates a standard 3D cube in FreeCAD, then identifies
    each of its six faces. It colors each face based on the provided 'colors'
    dictionary and saves images of the cube from eight different axonometric
    viewpoints.

    Args:
        colors (dict): A dictionary mapping face names (e.g., "Top", "Front")
                       to their RGB color tuples.
        save_folder (str): The path to the directory where the output will be saved.
        save_name (str): The name for the sample, used for the output folder
                         and file names.
    """
    doc = App.newDocument("colored_cube")
    cube = Part.makeBox(10, 10, 10)
    cube_obj = doc.addObject("Part::Feature", "Cube")
    cube_obj.Shape = cube

    faces = cube_obj.Shape.Faces

    min_z = float('inf')  
    max_z = -float('inf')  
    max_y = -float('inf') 
    min_y = float('inf') 
    min_x = float('inf')  
    max_x = -float('inf')  

    faces_idxs = {
        "Bottom": -1,  
        "Top": -1,  
        "Left": -1,  
        "Right": -1,  
        "Front": -1, 
        "Back": -1,  
    }
    for i, face in enumerate(faces):
        center = face.CenterOfMass
        if center.z < min_z:
            min_z = center.z
            faces_idxs["Bottom"] = i
        if center.z > max_z:
            max_z = center.z
            faces_idxs["Top"] = i
        if center.y > max_y:
            max_y = center.y
            faces_idxs["Back"] = i
        if center.y < min_y:
            min_y = center.y
            faces_idxs["Front"] = i
        if center.x < min_x:
            min_x = center.x
            faces_idxs["Left"] = i
        if center.x > max_x:
            max_x = center.x
            faces_idxs["Right"] = i
        
    default_color = (1.0, 1.0, 1.0)  
    cube_obj.ViewObject.DiffuseColor = [default_color] * len(faces)
    diffuse_colors = [default_color] * len(faces)
    for face_name in faces_idxs:
        face_index = faces_idxs[face_name]
        diffuse_colors[face_index] = colors[face_name]
        
    cube_obj.ViewObject.DiffuseColor = diffuse_colors
    cube_obj.ViewObject.DisplayMode = "Flat Lines"
    cube_obj.ViewObject.Transparency = 0 
    cube_obj.ViewObject.update()

    doc.recompute()
    
    # x: right=1, left=-1 
    # y: back=-1, front=1
    # z: top=1, bottom=-1
    views = {
        "FrontTopRight": (1, 1, 1),
        "FrontTopLeft": (-1, 1, 1),
        "FrontBottomRight": (1, -1, 1),
        "FrontBottomLeft": (-1, -1, 1),
        "BackTopRight": (1, 1, -1),
        "BackTopLeft": (-1, 1, -1),
        "BackBottomRight": (1, -1, -1),
        "BackBottomLeft": (-1, -1, -1),
    }

    time.sleep(1)
    view = FreeCADGui.ActiveDocument.ActiveView
    for view_name, direction in views.items():
        view.viewAxonometric()
        view.setViewDirection(direction)
        FreeCADGui.updateGui()
        view.fitAll()
        
        image_path = f"{save_folder}/{save_name}/{view_name}.png"
        view.saveImage(image_path, 1280, 1024) 
        time.sleep(1)
    
def save_json(matrix, faces, view_color, save_folder, save_name):
    """
    Saves cube net and face information to a JSON file.

    This function compiles the matrix representation of the cube net, the
    coordinates of each face, and their assigned colors into a JSON file
    for easy data access and reproducibility.

    Args:
        matrix (list of list of int): The 2D matrix representing the cube net.
        faces (dict): A dictionary mapping face names to their (row, column)
                      coordinates.
        view_color (dict): A dictionary mapping face names to their color names
                           (e.g., "red", "blue").
        save_folder (str): The path to the directory where the output will be saved.
        save_name (str): The name for the sample, used for the output file name.
    """
    json_data = {
        "matrix": matrix,
        "Top": {
            "position": faces["Top"], 
            "color": view_color["Top"]},
        "Bottom": {
            "position": faces["Bottom"], 
            "color": view_color["Bottom"]},
        "Left": {
            "position": faces["Left"], 
            "color": view_color["Left"]},
        "Right": {
            "position": faces["Right"], 
            "color": view_color["Right"]},
        "Front": {
            "position": faces["Front"], 
            "color": view_color["Front"]},
        "Back": {
            "position": faces["Back"], 
            "color": view_color["Back"]},
        "Axonometric": 
            [["Front", "Top", "Right"], 
             ["Front", "Top", "Left"],
             ["Front", "Bottom", "Right"],
             ["Front", "Bottom", "Left"],
             ["Back", "Top", "Right"],
             ["Back", "Top", "Left"],
             ["Back", "Bottom", "Right"],
             ["Back", "Bottom", "Left"]]
    }
    with open(f'{save_folder}/{save_name}/info.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False)

def main(args):
    """
    Main function to orchestrate the generation of cube nets and 3D models.
    
    This function initializes the cube net matrices and color definitions. It
    then iterates a specified number of times to generate different samples,
    each with a randomly selected net shape and color assignment. For each
    sample, it calls the helper functions to create the 2D net, the 3D cube,
    and the corresponding JSON data.
    
    Args:
        args (argparse.Namespace): The command-line arguments containing
                                   `save_folder` and `num_samples`.
    """
    save_folder = args.save_folder
    
    # Standard cube nets (unfolded)
    matrixs = [
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0]
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0]
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0]
        ],
        [
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]
        ],
        [
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0]
        ],
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0]
        ],
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]
        ],
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0]
        ],
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ],
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0]
        ]
    ]

    color_value = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "pink": (1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0)
    }

    face_names = ["Bottom", "Top", "Left", "Right", "Front", "Back"]
    colors = list(color_value.keys())

    for i in range(args.num_samples):
        save_name = f"sample_{i+1}"
        os.makedirs(f"{save_folder}/{save_name}", exist_ok=True)
        
        random.shuffle(colors)
        view_color_names = {k: v for k, v in zip(face_names, colors)}
        view_color_values = {k: color_value[v] for k, v in view_color_names.items()}
        
        matrix = random.choice(matrixs)
        
        try:
            base_row, base_col = 1, matrix[1].index(1)
            faces = identify_cube_faces(matrix, ("Bottom", base_row, base_col))
            create_squares(matrix, faces, view_color_values, save_folder, save_name)
            make_colored_cube(view_color_values, save_folder, save_name)
            save_json(matrix, faces, view_color_names, save_folder, save_name)
            print(f"Successfully generated sample {i+1} at {args.save_folder}/{save_name}")
        except Exception as e:
            print(f"An error occurred while generating sample {i+1}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate colored cube nets and 3D views.")
    parser.add_argument("--save_folder", type=str, required=True, 
                        help="Path to the directory where data will be saved, e.g., /path/to/save_folder")
    parser.add_argument("--num_samples", type=int, default=11,
                        help="Number of samples to generate (N).")
    args = parser.parse_args()
    main(args)

"""
    # run in FreeCAD
    import sys
    sys.argv = [
        "create_cube_reconstruction.py",
        "--save_folder", "/path/to/save",
        "--num_samples", "50"  
    ]
    exec(open("/path/to/create_cube_reconstruction.py").read())
"""