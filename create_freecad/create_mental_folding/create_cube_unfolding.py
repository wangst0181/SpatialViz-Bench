import os
import random
import json
import time
from collections import deque
import argparse
import Part
import FreeCADGui


def create_squares(matrix, faces, colors, save_folder, unit_size=1, save_name=None, mode=0, change_color=None):
    """
    Create unfolded cube squares based on a matrix and save the rendered image.

    Args:
        matrix (list): A 2D list representing the cube's unfolded pattern.
        faces (dict): A dictionary mapping face names (e.g., "Front", "Top") to their
                      (row, col) coordinates in the matrix.
        colors (dict): A dictionary mapping face names to their assigned colors.
        save_folder (str): The path to the folder where the images will be saved.
        unit_size (int, optional): The size of each square. Defaults to 1.
        save_name (str, optional): The name for the saved image file. Defaults to None.
        mode (int, optional): An integer used to differentiate between saved images. Defaults to 0.
        change_color (list, optional): A list of colors used to denote an incorrect
                                      unfolding. Defaults to None.
    """
    doc = App.newDocument("MatrixSquares")
    rows, cols = len(matrix), len(matrix[0])
    
    for face_name in faces:
        i, j = faces[face_name]
        x = j * unit_size
        y = (rows - 1 - i) * unit_size  
        square = Part.makePlane(unit_size, unit_size, App.Vector(x, y, 0))
        obj = doc.addObject("Part::Feature", f"Square_{i}_{j}")
        obj.Shape = square
        
        color = color_value[colors[face_name]]
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
    image_path = f"{save_folder}/{save_name}/{save_name}_unfold_correct_{mode}.png"
    if mode == -1:
        image_path = f"{save_folder}/{save_name}/{save_name}_unfold_incorrect_{'-'.join(change_color)}.png"
    view.saveImage(image_path, 1280, 1024)


def identify_cube_faces(matrix, base_face):
    """
    Identify cube face positions from a 2D matrix representing the unfolded net.

    Args:
        matrix (list): A 2D list representing the cube's unfolded pattern.
        base_face (tuple): A tuple containing the starting face name and its
                           (row, col) coordinates.

    Returns:
        dict: A dictionary mapping face names (e.g., "Front", "Top") to their
              corresponding (row, col) coordinates in the matrix.
    """
    faces = {}  
    visited = {}  
    
    face_name, r, c = base_face
    faces[face_name] = (r, c)
    
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
    Create a 3D cube with colored faces and save a rendered image from a random perspective.

    Args:
        colors (dict): A dictionary mapping face names to their assigned colors.
        save_folder (str): The path to the folder where the images will be saved.
        save_name (str): The name for the saved image file.

    Returns:
        str: The name of the selected view (e.g., "Front_Top_Right").
    """
    doc = App.newDocument("colored_cube")
    cube = Part.makeBox(10, 10, 10)
    cube_obj = doc.addObject("Part::Feature", "Cube")
    cube_obj.Shape = cube

    faces = cube_obj.Shape.Faces

    min_z, max_z = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    min_x, max_x = float('inf'), -float('inf')

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
        diffuse_colors[face_index] = color_value[colors[face_name]]
        
    cube_obj.ViewObject.DiffuseColor = diffuse_colors
    cube_obj.ViewObject.DisplayMode = "Flat Lines"
    cube_obj.ViewObject.Transparency = 0 
    cube_obj.ViewObject.update()

    doc.recompute()
    FreeCADGui.SendMsgToActiveView("ViewFit")
    
    views = {
        "Front_Top_Right": (-1, 1, 1),
        "Front_Top_Left": (1, 1, -1),
        "Front_Bottom_Right": (-1, 1, 1),
        "Front_Bottom_Left": (1, 1, 1),
        "Back_Top_Right": (-1, -1, -1),
        "Back_Top_Left": (1, -1, -1),
        "Back_Bottom_Right": (-1, -1, 1),
        "Back_Bottom_Left": (1, -1, 1),
    }
        
    view_selected = random.choice(list(views.keys()))
    direction = views[view_selected]
    
    time.sleep(1)
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewAxonometric()
    FreeCADGui.updateGui()
    view.fitAll()
    time.sleep(1)
    
    if view_selected != "Front_Top_Right":
        view = FreeCADGui.ActiveDocument.ActiveView
        view.setViewDirection(direction)
        FreeCADGui.updateGui()
        view.fitAll()
        time.sleep(1)
    
    faces = view_selected.split('_')
    faces_color = [colors[face_name] for face_name in faces]
    
    image_path = f"{save_folder}/{save_name}/{'_'.join(faces_color)}_cube.png"
    view.saveImage(image_path, 1280, 1024) 
    return view_selected


def main(args):
    """
    Main function to generate a cube unfolding dataset.

    Args:
        args (argparse.Namespace): Command-line arguments containing
                                   `save_folder` and `num_samples`.
    """
    save_folder = args.save_folder
    num_samples = args.num_samples

    os.makedirs(save_folder, exist_ok=True)

    matrixs = [
        [[1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [0, 1, 0, 0, 0]],
        [[1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [0, 0, 1, 0, 0]],
        [[1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [0, 0, 0, 1, 0]],
        [[0, 1, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [0, 0, 1, 0, 0]],
        [[0, 1, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [0, 1, 0, 0, 0]],
        [[1, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0]],
        [[1, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0]],
        [[1, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0]],
        [[1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0]],
        [[1, 1, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 1, 0]]
    ]

    for i in range(num_samples):
        save_name = i
        os.makedirs(f"{save_folder}/{save_name}", exist_ok=True)
        
        view_oppo = {"Bottom": "Top", "Top": "Bottom", "Left": "Right", "Right": "Left", "Front": "Back", "Back": "Front"}
        
        global color_value
        color_value = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "pink": (1.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0)
        }
        colors = list(color_value.keys())
        random.shuffle(colors)
        view_color = {k: v for k, v in zip(list(view_oppo.keys()), colors)}

        matrixs_selected = random.sample(matrixs, 3)
        view = make_colored_cube(view_color, save_folder, save_name)
        views = view.split('_')
        
        for k, matrix in enumerate(matrixs_selected):
            base_row, base_col = 1, matrix[1].index(1)
            faces = identify_cube_faces(matrix, ("Bottom", base_row, base_col))
            create_squares(matrix, faces, view_color, save_folder, 1, save_name, mode=k)
        
        incorrect_matrix = random.choice(matrixs)
        views_selected = random.sample(views, 2)
        view_0 = views_selected[0]
        view_0_oppo = view_oppo[view_0]
        view_1 = views_selected[1]
        view_color[view_1], view_color[view_0_oppo] = view_color[view_0_oppo], view_color[view_1]
        
        base_row, base_col = 1, incorrect_matrix[1].index(1)
        faces = identify_cube_faces(incorrect_matrix, ("Bottom", base_row, base_col))
        create_squares(incorrect_matrix, faces, view_color, save_folder, 1, save_name, mode=-1, change_color=[view_color[view_1], view_color[view_0_oppo]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cube unfolding dataset")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to save results, e.g. /path/to/save")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    args = parser.parse_args()
    main(args)

"""
    # run in FreeCAD
    import sys

    sys.argv = [
        "create_cube_unfolding.py",
        "--save_folder", "/path/to/save",
        "--num_samples", "50"
    ]

    exec(open("/path/to/create_cube_unfolding.py").read())

"""