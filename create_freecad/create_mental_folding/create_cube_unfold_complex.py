import os
import sys
import time
import json
import random
import argparse
from collections import deque

import Part
import FreeCADGui


def identify_cube_faces(matrix, base_face):
    """
    Identifies the positions of each cube face on an unfolded net represented by a matrix.

    This function uses a Breadth-First Search (BFS) approach starting from a known base face.
    It navigates the matrix to find all connected '1's, which represent the faces, and maps
    them to their corresponding cube face names (e.g., 'Top', 'Bottom', 'Front').

    Args:
        matrix (list of lists): A 2D list representing the unfolded cube net, where '1's
                                 are faces and '0's are empty spaces.
        base_face (tuple): A tuple containing the name of the base face (str), its row index (int),
                           and its column index (int) in the matrix.

    Returns:
        dict: A dictionary mapping cube face names to their (row, column) coordinates in the matrix.
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
                if (
                    0 <= nr < len(matrix)
                    and 0 <= nc < len(matrix[0])
                    and matrix[nr][nc] == 1
                    and (nr, nc) not in visited
                ):
                    faces[new_face] = (nr, nc)
                    queue.append((new_face, nr, nc))
                    visited[(nr, nc)] = new_face
    return faces


def create_squares(matrix, faces, images, plane_name, unit_size=10,
                   save_name=None, mode=0, change_pattern=None,
                   asymmetric=['2', '3', '6'], k=0):
    """
    Creates an unfolded cube net with images on each face and saves it as an image.

    This function leverages FreeCAD's ImagePlane objects to represent the faces of the
    unfolded net. It places these planes according to the provided matrix and face
    positions, applies the specified images, and handles rotations for different net
    configurations. It then captures and saves an image of the resulting FreeCAD view.

    Args:
        matrix (list of lists): The matrix representation of the unfolded net.
        faces (dict): A dictionary mapping face names to their (row, col) coordinates.
        images (dict): A dictionary mapping face names to the image names (patterns) to be applied.
        plane_name (str): The name of the predefined plane configuration.
        unit_size (int, optional): The size of each square face in FreeCAD units. Defaults to 10.
        save_name (str, optional): The name to use for the output folder and file. Defaults to None.
        mode (int, optional): Controls the type of output image.
                              0: Correct unfold.
                              -1: Incorrect unfold (swapped images).
                              -2: Incorrect unfold (rotated asymmetric images).
                              Defaults to 0.
        change_pattern (list, optional): A list of two image names that were swapped in mode -1.
                                         Defaults to None.
        asymmetric (list, optional): A list of image names considered "asymmetric".
                                     Used in mode -2 to rotate images. Defaults to ['2', '3', '6'].
        k (int, optional): A counter used to generate unique filenames. Defaults to 0.

    Returns:
        list: A list of face names that had their images rotated in mode -2.
    """
    doc = App.newDocument("MatrixSquares")
    rows, cols = len(matrix), len(matrix[0])
    choice_patterns = random.sample(asymmetric, random.randint(1, 3))
    choice_patterns_face = []
    for face_name in faces:
        i, j = faces[face_name]
        x = j * unit_size
        y = (rows - 1 - i) * unit_size

        plane = doc.addObject("Image::ImagePlane", f"Image{face_name}")
        if mode == -2:
            if images[face_name] in choice_patterns:
                plane.ImageFile = image_paths[f"{images[face_name]}_0"]
                choice_patterns_face.append(face_name)
            else:
                plane.ImageFile = image_paths[images[face_name]]
        else:
            plane.ImageFile = image_paths[images[face_name]]

        plane.XSize = unit_size
        plane.YSize = unit_size
        plane.Placement.Base = App.Vector(x + unit_size / 2, y + unit_size / 2, 0)
        if plane_name != "planes_1_4_1_0":
            if face_name in info[plane_name]["planes_info"]:
                rotation = info[plane_name]["planes_info"][face_name]
                plane.Placement.Rotation = rotation

    FreeCADGui.runCommand("Std_DrawStyle", 4)
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewIsometric()
    FreeCADGui.updateGui()
    view.fitAll()
    image_path = f"{save_folder}/{save_name}/{save_name}_unfold_correct_{k}.png"
    if mode == -1:
        image_path = f"{save_folder}/{save_name}/{save_name}_unfold_incorrect_change_{'-'.join(change_pattern)}.png"
    elif mode == -2:
        image_path = f"{save_folder}/{save_name}/{save_name}_unfold_incorrect_{k}_transform_{'-'.join(choice_patterns)}.png"
    view.saveImage(image_path, 1280, 1024)
    return choice_patterns_face


def make_cube_wi_images(images, plane_name, view_selected, save_name):
    """
    Creates a folded cube with images on its faces and saves it as an image.

    This function creates a cube in FreeCAD by placing six ImagePlane objects,
    each corresponding to a face of the cube. It applies the specified image
    textures to each plane and orients the cube to a specific viewpoint.
    Finally, it captures and saves an image of the cube.

    Args:
        images (dict): A dictionary mapping face names to the image names (patterns) to be applied.
        plane_name (str): The name of the predefined plane configuration for the cube.
        view_selected (str): The name of the desired isometric view (e.g., 'Front_Top_Right').
        save_name (str): The name to use for the output folder and file.
    """
    doc = App.newDocument("colored_cube")

    for face_name, image_name in images.items():
        plane = doc.addObject("Image::ImagePlane", f"Image{face_name}-{image_name}")
        plane.ImageFile = image_paths[image_name]
        plane.XSize = cube_size
        plane.YSize = cube_size
        plane.Placement = info[plane_name]["planes_info"][face_name]

    FreeCADGui.runCommand("Std_DrawStyle", 4)

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

    faces = view_selected.split("_")
    faces_pattern = [images[face_name] for face_name in faces]

    image_path = f"{save_folder}/{save_name}/{'_'.join(faces_pattern)}_cube.png"
    view.saveImage(image_path, 1280, 1024)

def find_most_matching_view(views, face_set):
    """
    Finds the cube view that best matches a given set of visible faces.

    This function calculates the number of common faces between a set of
    visible faces on the cube and the faces defined for each predefined view.
    It returns the name of the view with the highest number of matches.

    Args:
        views (dict): A dictionary mapping view names to their directional vectors.
        face_set (set): A set of cube face names (e.g., 'Front', 'Top', 'Right').

    Returns:
        str: The name of the view that has the most faces in common with the face_set.
    """
    def match_count(key):
        key_faces = set(key.split("_"))
        return len(key_faces & face_set)

    return max(views.keys(), key=match_count)


def main(args):
    """
    Main function to generate cube and unfolding images.

    This function orchestrates the entire image generation process. It sets up global
    variables, defines cube net matrices and placements, and then enters a loop to
    generate a specified number of samples. For each sample, it:
    1.  Assigns random images to each cube face.
    2.  Generates and saves images of two different correct unfolded nets.
    3.  Generates an image of a cube from a view that best matches a set of transformed faces.
    4.  Generates an image of an incorrect unfolded net with swapped images.

    Args:
        args (argparse.Namespace): Command-line arguments containing paths and sample count.
    """
    global save_folder, image_paths, cube_size, info, views
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    # You should prepare the images in advance
    # Define image paths for patterns
    # _0 suffix indicates a rotated version of the base image
    image_paths = {
        "1": os.path.join(args.image_folder, "1.png"),
        "1_0": os.path.join(args.image_folder, "1_0.png"),
        "2": os.path.join(args.image_folder, "2.png"),
        "2_0": os.path.join(args.image_folder, "2_0.png"),
        "3": os.path.join(args.image_folder, "3.png"),
        "3_0": os.path.join(args.image_folder, "3_0.png"),
        "4": os.path.join(args.image_folder, "4.png"),
        "4_0": os.path.join(args.image_folder, "4_0.png"),
        "5": os.path.join(args.image_folder, "5.png"),
        "5_0": os.path.join(args.image_folder, "5_0.png"),
        "6": os.path.join(args.image_folder, "6.png"),
        "6_0": os.path.join(args.image_folder, "6_0.png"),
    }

    cube_size = 10

    # pivot
    # [
    #     [1, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 0],
    #     [1, 0, 0, 0, 0]
    # ]
    planes_1_4_1_0 = {
        "Top": App.Placement(
            App.Vector(cube_size/2, cube_size/2, cube_size),
            App.Rotation(App.Vector(0, 1, 0), 180)
        ),
        "Bottom": App.Placement(
            App.Vector(cube_size/2, cube_size/2, 0),
            App.Rotation(App.Vector(1, 0, 0), 0)
        ),  # pivot

        "Right": App.Placement(
            App.Vector(cube_size, cube_size/2, cube_size/2),
            App.Rotation(App.Vector(0, 1, 0), -90)
        ),
        "Left": App.Placement(
            App.Vector(0, cube_size/2, cube_size/2),
            App.Rotation(App.Vector(0, 1, 0), 90)
        ),
        "Front": App.Placement(
            App.Vector(cube_size/2, 0, cube_size/2),
            App.Rotation(App.Vector(1, 0, 0), -90)
        ),
        "Back": App.Placement(
            App.Vector(cube_size/2, cube_size, cube_size/2),
            App.Rotation(App.Vector(1, 0, 0), 90)
        ),
    }


    # [
    #     [1, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 0],
    #     [0, 1, 0, 0, 0]
    # ]

    planes_1_4_1_1 = {
        "Front": App.Rotation(App.Vector(0, 0, 1), 90) 
    }


    # [
    #     [1, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 0],
    #     [0, 0, 1, 0, 0]
    # ]

    planes_1_4_1_2 = {
        "Front": App.Rotation(App.Vector(0, 0, 1), -180)
    }

    # [
    #     [1, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 0],
    #     [0, 0, 0, 1, 0]
    # ]

    planes_1_4_1_3 = {
        "Front": App.Rotation(App.Vector(0, 0, 1), -90)
    }

    # [
    #     [0, 1, 0, 0, 0],
    #     [1, 1, 1, 1, 0],
    #     [0, 1, 0, 0, 0]
    # ]

    planes_1_4_1_4 = {
        "Front": App.Rotation(App.Vector(0, 0, 1), 90),
        "Back": App.Rotation(App.Vector(0, 0, 1), -90)
    }

    # [
    #     [0, 1, 0, 0, 0],
    #     [1, 1, 1, 1, 0],
    #     [0, 0, 1, 0, 0]
    # ]

    planes_1_4_1_5 = {
        "Front": App.Rotation(App.Vector(0, 0, 1), -180),
        "Back": App.Rotation(App.Vector(0, 0, 1), -90),
    }


    # [
    #     [1, 1, 0, 0, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 1, 0, 0, 0]
    # ]

    planes_2_3_1_0 = {
        "Left": App.Rotation(App.Vector(0, 0, 1), -90) 
    }

    # [
    #     [1, 1, 0, 0, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 0, 1, 0, 0]
    # ]

    planes_2_3_1_1 = {
        "Left": App.Rotation(App.Vector(0, 0, 1), -90),
        "Front": App.Rotation(App.Vector(0, 0, 1), 90)
    }

    # [
    #     [1, 1, 0, 0, 0],
    #     [0, 1, 1, 1, 0],
    #     [0, 0, 0, 1, 0]
    # ]

    planes_2_3_1_2 = {
        "Left": App.Rotation(App.Vector(0, 0, 1), -90),
        "Front": App.Rotation(App.Vector(0, 0, 1), -180)
    }

    # [
    #     [1, 1, 1, 0, 0],
    #     [0, 0, 1, 1, 1],
    #     [0, 0, 0, 0, 0]
    # ]

    planes_3_3 = {
        "Left": App.Rotation(App.Vector(0, 0, 1), -90),
        "Front": App.Rotation(App.Vector(0, 0, 1), -180)
    }

    # [
    #     [1, 1, 0, 0, 0],
    #     [0, 1, 1, 0, 0],
    #     [0, 0, 1, 1, 0]
    # ]
    planes_2_2_2 = {
        "Top":  App.Rotation(App.Vector(0, 0, 1), -90),
        "Left": App.Rotation(App.Vector(0, 0, 1), -90),
        "Front": App.Rotation(App.Vector(0, 0, 1), 90)
    }
    
    info = {
        "planes_1_4_1_0": 
            {
                "planes_info": planes_1_4_1_0, 
                "matrix": [
                            [1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0],
                            [1, 0, 0, 0, 0]
                        ]
            }, 
        "planes_1_4_1_1": 
            {
                "planes_info": planes_1_4_1_1, 
                "matrix": [
                            [1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0],
                            [0, 1, 0, 0, 0]
                        ]
            },
        "planes_1_4_1_2": 
            {
                "planes_info": planes_1_4_1_2, 
                "matrix": [
                            [1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]
                        ]
            },
        "planes_1_4_1_3": 
            {
                "planes_info": planes_1_4_1_3, 
                "matrix": [
                            [1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0],
                            [0, 0, 0, 1, 0]
                        ]
            },
        "planes_1_4_1_4": 
            {
                "planes_info": planes_1_4_1_4, 
                "matrix": [
                            [0, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0],
                            [0, 1, 0, 0, 0]
                        ]
            },
        "planes_1_4_1_5": 
            {
                "planes_info": planes_1_4_1_5, 
                "matrix": [
                            [0, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]
                        ]
            },
        "planes_2_3_1_0": 
            {
                "planes_info": planes_2_3_1_0, 
                "matrix": [
                            [1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 0, 0, 0]
                        ]
            },
        "planes_2_3_1_1": 
            {
                "planes_info": planes_2_3_1_1, 
                "matrix": [
                            [1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]
                        ]
            },
        "planes_2_3_1_2":
            {
                "planes_info": planes_2_3_1_2, 
                "matrix": [
                            [1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 1, 0]
                        ]
            },
        "planes_3_3": 
            {
                "planes_info": planes_3_3, 
                "matrix": [
                            [1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1],
                            [1, 0, 0, 0, 0]
                        ]
            },
        "planes_2_2_2": 
            {
                "planes_info": planes_2_2_2, 
                "matrix": [
                            [1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0]
                        ]
            },
    }

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

    for i in range(args.num_samples):
        save_name = i
        os.makedirs(f"{save_folder}/{save_name}", exist_ok=True)

        view_oppo = {
            "Bottom": "Top",
            "Top": "Bottom",
            "Left": "Right",
            "Right": "Left",
            "Front": "Back",
            "Back": "Front",
        }

        images = ["1", "2", "3", "4", "5", "6"]
        random.shuffle(images)
        view_image = {k: v for k, v in zip(list(view_oppo.keys()), images)}

        plane_matrixs = []
        for key, value in info.items():
            matrix = value["matrix"]
            plane_matrixs.append({"matrix": matrix, "plane_name": key})
        plane_matrixs_selected = random.sample(plane_matrixs, 2)

        choice_patterns_face_set = set()
        for k, item in enumerate(plane_matrixs_selected):
            matrix = item["matrix"]
            plane_name = item["plane_name"]
            base_row, base_col = 1, matrix[1].index(1)
            faces = identify_cube_faces(matrix, ("Bottom", base_row, base_col))
            create_squares(matrix, faces, view_image, plane_name, 1, save_name, mode=0, k=k)

            choice_patterns_face = create_squares(
                matrix,
                faces,
                view_image,
                plane_name,
                1,
                save_name,
                mode=-2,
                asymmetric=["1", "2", "3", "4", "5", "6"],
                k=k,
            )
            choice_patterns_face_set.update(choice_patterns_face)

        view_selected = find_most_matching_view(views, choice_patterns_face_set)
        make_cube_wi_images(view_image, "planes_1_4_1_0", view_selected, save_name)
        faces_selected = view_selected.split("_")

        fixed_mapping = {k: view_image[k] for k in faces}
        remaining_views = [k for k in view_image if k not in faces]
        remaining_images = [view_image[k] for k in view_image if k not in faces]
        random.shuffle(remaining_images)
        random_mapping = {k: v for k, v in zip(remaining_views, remaining_images)}
        view_image_new = {**fixed_mapping, **random_mapping}

        plane_matrixs_selected = random.sample(plane_matrixs, 2)
        for k, item in enumerate(plane_matrixs_selected):
            matrix = item["matrix"]
            plane_name = item["plane_name"]
            base_row, base_col = 1, matrix[1].index(1)
            faces = identify_cube_faces(matrix, ("Bottom", base_row, base_col))
            create_squares(matrix, faces, view_image_new, plane_name, 1, save_name, mode=0, k=k + 2)

        incorrect_item = random.choice(plane_matrixs)
        incorrect_matrix = incorrect_item["matrix"]
        incorrect_plane_name = incorrect_item["plane_name"]
        faces_selected = random.sample(faces_selected, 2)
        view_0 = faces_selected[0]
        view_0_oppo = view_oppo[view_0]
        view_1 = faces_selected[1]
        view_image[view_1], view_image[view_0_oppo] = (
            view_image[view_0_oppo],
            view_image[view_1],
        )

        base_row, base_col = 1, incorrect_matrix[1].index(1)
        faces = identify_cube_faces(incorrect_matrix, ("Bottom", base_row, base_col))
        create_squares(
            incorrect_matrix,
            faces,
            view_image,
            incorrect_plane_name,
            1,
            save_name,
            mode=-1,
            change_pattern=[view_image[view_1], view_image[view_0_oppo]],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, required=True, help="Path to save results")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to cube face images")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    args = parser.parse_args()
    main(args)

"""
    # run in FreeCAD
    import sys
    sys.argv = [
        "create_cube_unfolding_complex.py",
        "--save_folder", "/path/to/save",
        "--image_folder", "/path/to/images"
        "--num_samples", "50",
    ]
    exec(open("/path/to/create_cube_unfolding_complex.py").read())
"""