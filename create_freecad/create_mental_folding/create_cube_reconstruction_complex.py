from collections import deque
import FreeCAD as App
import FreeCADGui as Gui
import random, json, os, time
import argparse


def identify_cube_faces(matrix, base_face):
    """
    Identifies the positions of each cube face on the 2D net using Breadth-First Search (BFS).

    This function traverses the 2D cube net matrix, which is represented by 1s and 0s, starting from a given base face. 
    It uses a BFS approach to explore adjacent faces and assigns a specific name (e.g., 'Top', 'Bottom', 'Front') 
    to each face's coordinates based on their relative position to each other.

    Args:
        matrix (list): The 2D matrix representing the cube net, where 1s denote a face and 0s are empty spaces.
        base_face (tuple): A tuple containing the starting face's name and its (row, column) coordinates, e.g., ("Bottom", 1, 1).

    Returns:
        dict: A dictionary mapping face names (strings) to their (row, column) coordinates (tuples).
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
        if current_face not in adjacent_moves:
            continue
        for new_face, (dr, dc) in adjacent_moves[current_face]:
            nr, nc = r + dr, c + dc
            if new_face not in faces:
                if 0 <= nr < len(matrix) and 0 <= nc < len(matrix[0]) and matrix[nr][nc] == 1 and (nr, nc) not in visited:
                    faces[new_face] = (nr, nc)
                    queue.append((new_face, nr, nc))
                    visited[(nr, nc)] = new_face
    return faces


def create_squares(matrix, faces, image_paths, images, unit_size=10, save_name=None, save_folder=None):
    """
    Creates a 2D representation of the cube net with image textures in FreeCAD and saves it as an image.

    This function generates a 2D FreeCAD document, creates a separate ImagePlane for each face of the cube net, 
    and applies the specified image textures. It then adjusts the view and saves an isometric image of the net.

    Args:
        matrix (list): The 2D matrix of the cube net used to determine dimensions.
        faces (dict): A dictionary mapping face names to their coordinates on the net.
        image_paths (dict): A dictionary mapping image identifiers to their file paths.
        images (dict): A dictionary mapping cube face names to the image identifiers to be used.
        unit_size (int, optional): The size of each square face in the FreeCAD document. Defaults to 10.
        save_name (str, optional): The name to be used for the output file. Defaults to None.
        save_folder (str, optional): The directory where the image will be saved. Defaults to None.
    """
    doc = App.newDocument("MatrixSquares")
    rows, cols = len(matrix), len(matrix[0])

    for face_name in faces:
        i, j = faces[face_name]
        x = j * unit_size
        y = (rows - 1 - i) * unit_size
        
        plane = doc.addObject("Image::ImagePlane", f"Image{face_name}")
        plane.ImageFile = image_paths[images[face_name]]
        plane.XSize = unit_size
        plane.YSize = unit_size
        plane.Placement.Base = App.Vector(x + unit_size / 2, y + unit_size / 2, 0)
        
    FreeCADGui.runCommand('Std_DrawStyle', 4)
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewIsometric()
    FreeCADGui.updateGui()
    view.fitAll()
    image_path = os.path.join(save_folder, str(save_name), f"{save_name}_unfold.png")
    view.saveImage(image_path, 1280, 1024)
    App.closeDocument("MatrixSquares")


def make_cube_wi_images(images, plane_name, info, image_paths, save_folder, save_name, cube_size, mode=0, asymmetric=['2', '3', '6']):
    """
    Creates a 3D cube with image textures and saves various isometric views.

    This function generates a 3D FreeCAD document of a cube, applies different image textures to each face, 
    and then renders and saves images from multiple predefined isometric viewpoints. It can also create a 'mode 1' 
    version where certain asymmetric images are rotated to create an "incorrect" example.

    Args:
        images (dict): A dictionary mapping cube face names to the image identifiers to be used.
        plane_name (str): The name of the predefined cube net pattern (e.g., "planes_1_4_1_0").
        info (dict): A dictionary containing placement and matrix information for different cube nets.
        image_paths (dict): A dictionary mapping image identifiers to their file paths.
        save_folder (str): The directory where the images will be saved.
        save_name (str): The name to be used for the output file.
        cube_size (int): The side length of the cube in FreeCAD units.
        mode (int, optional): Controls the image application. 0 for correct orientation, 1 for incorrect (rotated) orientation. Defaults to 0.
        asymmetric (list, optional): A list of image identifiers considered asymmetric and can be rotated in mode 1. Defaults to ['2', '3', '6'].
    """
    doc = App.newDocument("colored_cube")
    choice_patterns = random.sample(asymmetric, random.randint(1, len(asymmetric)))
    choice_patterns_face = []
    
    for face_name, image_name in images.items():
        plane = doc.addObject("Image::ImagePlane", f"Image{face_name}-{image_name}")
        if mode == 0:
            plane.ImageFile = image_paths[image_name]
        elif mode == 1:
            if image_name in choice_patterns:
                plane.ImageFile = image_paths[f"{image_name}_0"]
                choice_patterns_face.append(face_name)
            else:
                plane.ImageFile = image_paths[image_name]
        plane.XSize = cube_size
        plane.YSize = cube_size
        plane.Placement = info[plane_name]["planes_info"][face_name]

    FreeCADGui.runCommand('Std_DrawStyle', 4)
    
    views = {
        "FrontTopRight": (-1, 1, 1),
        "FrontTopLeft": (1, 1, -1),
        "FrontBottomRight": (-1, 1, 1),
        "FrontBottomLeft": (1, 1, 1),
        "BackTopRight": (-1, -1, -1),
        "BackTopLeft": (1, -1, -1),
        "BackBottomRight": (-1, -1, 1),
        "BackBottomLeft": (1, -1, 1),
    }

    views_selected = random.sample(list(views), 3)
    
    time.sleep(1)
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewAxonometric()
    FreeCADGui.updateGui()
    view.fitAll()
    time.sleep(1)
    
    for view_name in views_selected:
        direction = views[view_name]
        view.setViewDirection(direction)
        FreeCADGui.updateGui()
        view.fitAll()
        time.sleep(1)
        
        image_path = os.path.join(save_folder, str(save_name), f"{view_name}.png")
        if mode == 1:
            image_path = os.path.join(save_folder, str(save_name), f"{view_name}_incorrect_{'-'.join(choice_patterns)}.png")
        view.saveImage(image_path, 1280, 1024)
        time.sleep(1)
    
    App.closeDocument("colored_cube")


def save_json(matrix, faces, view_image, image_paths, save_folder, save_name):
    """
    Saves the cube net and image information to a JSON file.

    This function compiles all the relevant data, including the cube net matrix, the position of each face, 
    the image applied to each face, and a list of possible views, into a structured JSON format. This file 
    serves as a record of the generated data.

    Args:
        matrix (list): The 2D matrix of the cube net.
        faces (dict): A dictionary mapping face names to their coordinates.
        view_image (dict): A dictionary mapping face names to the image identifiers.
        image_paths (dict): A dictionary mapping image identifiers to their file paths.
        save_folder (str): The directory where the JSON file will be saved.
        save_name (str): The name to be used for the output file.
    """
    json_data = {
        "matrix": matrix,
        "Top": {
            "position": faces.get("Top"),
            "image_path": image_paths.get(view_image.get("Top"))},
        "Bottom": {
            "position": faces.get("Bottom"),
            "image_path": image_paths.get(view_image.get("Bottom"))},
        "Left": {
            "position": faces.get("Left"),
            "image_path": image_paths.get(view_image.get("Left"))},
        "Right": {
            "position": faces.get("Right"),
            "image_path": image_paths.get(view_image.get("Right"))},
        "Front": {
            "position": faces.get("Front"),
            "image_path": image_paths.get(view_image.get("Front"))},
        "Back": {
            "position": faces.get("Back"),
            "image_path": image_paths.get(view_image.get("Back"))},
        "Axonometric":
            [["Back", "Bottom", "Left"],
             ["Back", "Bottom", "Right"],
             ["Back", "Top", "Left"],
             ["Back", "Top", "Right"],
             ["Front", "Bottom", "Left"],
             ["Front", "Bottom", "Right"],
             ["Front", "Top", "Left"],
             ["Front", "Top", "Right"]]
    }
    file_path = os.path.join(save_folder, str(save_name), "info.json")
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False)


def main(args):
    """
    Main function to run the data generation process.

    This function orchestrates the entire process: it reads command-line arguments, selects a random cube net, 
    identifies the face positions on that net, generates the 2D and 3D models with images, and saves all 
    the generated files (images and JSON) to the specified directories.

    Args:
        args: An object containing command-line arguments, including save_folder, image_folder, and num_samples.
    """
    cube_size = 10
    
    # You should prepare the images in advance
    # Define image paths for patterns
    # _0 suffix indicates a rotated version of the base image
    image_paths = {
        "1": os.path.join(args.image_folder, "1.png"),     
        "2": os.path.join(args.image_folder, "2.png"),  
        "2_0":  os.path.join(args.image_folder, "1.png"),     
        "3": os.path.join(args.image_folder, "3.png"), 
        "3_0":  os.path.join(args.image_folder, "4.png"),   
        "4": os.path.join(args.image_folder, "4.png"),    
        "5": os.path.join(args.image_folder, "5.png"),    
        "6": os.path.join(args.image_folder, "6.png"),
        "6_0":  os.path.join(args.image_folder, "6_0.png"),   
    }

    # Cube net placement information
    info = {
        "planes_1_4_1_0": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0]]
        },
        "planes_1_4_1_1": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), -90)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 1, 0, 0, 0]]
        },
        "planes_1_4_1_2": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), 180)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 1, 0, 0]]
        },
        "planes_1_4_1_3": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), 90)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0]]
        },
        "planes_1_4_1_4": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), -90)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90) * App.Rotation(App.Vector(0, 0, 1), 90)),
            },
            "matrix": [[0, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 1, 0, 0, 0]]
        },
        "planes_1_4_1_5": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), 180)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90) * App.Rotation(App.Vector(0, 0, 1), 90)),
            },
            "matrix": [[0, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 1, 0, 0]]
        },
        "planes_2_3_1_0": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90) * App.Rotation(App.Vector(0, 0, 1), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 0]]
        },
        "planes_2_3_1_1": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90) * App.Rotation(App.Vector(0, 0, 1), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), -90)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]
        },
        "planes_2_3_1_2": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90) * App.Rotation(App.Vector(0, 0, 1), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), 180)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0]]
        },
        "planes_3_3": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(0, 1, 0), 180)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90) * App.Rotation(App.Vector(0, 0, 1), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), 180)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]]
        },
        "planes_2_2_2": {
            "planes_info": {
                "Top": App.Placement(App.Vector(cube_size/2, cube_size/2, cube_size), App.Rotation(App.Vector(1, 0, 0), 180) * App.Rotation(App.Vector(0, 0, 1), -90)),
                "Bottom": App.Placement(App.Vector(cube_size/2, cube_size/2, 0), App.Rotation(App.Vector(1, 0, 0), 0)),
                "Right": App.Placement(App.Vector(cube_size, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), -90)),
                "Left": App.Placement(App.Vector(0, cube_size/2, cube_size/2), App.Rotation(App.Vector(0, 1, 0), 90) * App.Rotation(App.Vector(0, 0, 1), 90)),
                "Front": App.Placement(App.Vector(cube_size/2, 0, cube_size/2), App.Rotation(App.Vector(1, 0, 0), -90) * App.Rotation(App.Vector(0, 0, 1), -90)),
                "Back": App.Placement(App.Vector(cube_size/2, cube_size, cube_size/2), App.Rotation(App.Vector(1, 0, 0), 90)),
            },
            "matrix": [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]]
        },
    }

    matrixs = [item["matrix"] for item in info.values()]

    for i in range(args.num_samples):
        save_name = i + 1
        sample_folder = os.path.join(args.save_folder, str(save_name))
        
        try:
            os.makedirs(sample_folder, exist_ok=True)
            
            view_name = ["Bottom", "Top", "Left", "Right", "Front", "Back"]
            images_list = ['1', '2', '3', '4', '5', '6']
            random.shuffle(images_list)
            view_image = {k: v for k, v in zip(view_name, images_list)}
            
            matrix = random.choice(matrixs)
            
            plane_name = ""
            for name, value in info.items():
                if value["matrix"] == matrix:
                    plane_name = name
                    break
            
            base_row, base_col = -1, -1
            for r_idx, row in enumerate(matrix):
                if 1 in row:
                    base_row, base_col = r_idx, row.index(1)
                    break
            
            if base_row == -1:
                print(f"Skipping sample {save_name}: Invalid matrix.")
                continue
                
            faces = identify_cube_faces(matrix, ("Bottom", base_row, base_col))
            
            if len(faces) == 6:
                print(f"Generating sample {save_name}...")
                create_squares(matrix, faces, image_paths, view_image, 10, save_name, args.save_folder)
                make_cube_wi_images(view_image, plane_name, info, image_paths, args.save_folder, save_name, cube_size, mode=0)
                make_cube_wi_images(view_image, plane_name, info, image_paths, args.save_folder, save_name, cube_size, mode=1, asymmetric=['2', '3', '6'])
                save_json(matrix, faces, view_image, image_paths, args.save_folder, save_name)
            else:
                print(f"Skipping sample {save_name}: Could not identify all 6 faces from the chosen net.")
                continue

        except Exception as e:
            print(f"An error occurred while processing sample {save_name}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate cube folding and view data with images.')
    parser.add_argument('--save_folder', type=str, required=True,
                        help='Path to the directory where the data will be saved, e.g., /path/to/save')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to the directory containing image files, e.g., /path/to/pics')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate.')
    
    args = parser.parse_args() 
    main(args)
    
"""
    import sys
    sys.argv = [
        "create_cube_reconstruction_complex.py",
        "--save_folder", "/path/to/save",
        "--image_folder", "/path/to/images",
        "--num_samples", "50"  
    ]
    exec(open("/path/to/create_cube_reconstruction_complex.py").read())
"""