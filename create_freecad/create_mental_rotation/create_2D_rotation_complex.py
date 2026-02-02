import os
import random
import argparse
import Part
import FreeCADGui
import time

def create_squares(matrix, save_dir, image_paths, unit_size=10, marker_size=2, save_name=None, mode=0, asymmetric=['2', '3', '6'], record=[]):
    """
    Creates a grid of squares in FreeCAD from a binary matrix, replacing '1's with images.
    
    This function generates a 2D grid based on a binary matrix. For each cell with a value of 1,
    it places an image plane using a randomly selected pattern image. In `mode=1`, it applies
    a predefined asymmetric transformation to a subset of the patterns to create distorted
    versions. A red marker is added at a random corner for orientation, and the final
    image is saved as a PNG.

    Args:
        matrix (list of list): A 2D binary matrix where `1` indicates an image should be placed.
        save_dir (str): The directory to save the output image.
        image_paths (dict): A dictionary mapping pattern names to their image file paths.
        unit_size (int, optional): The side length of each square/image plane. Defaults to 10.
        marker_size (int, optional): The size of the red corner marker. Defaults to 2.
        save_name (str, optional): The base name for the output file. Defaults to None.
        mode (int, optional): If `1`, enables asymmetric transformation mode. Defaults to 0.
        asymmetric (list, optional): A list of pattern names that have asymmetric versions. Defaults to ['2', '3', '6'].
        record (list, optional): A list used to store the sequence of images and the marker for reproducibility. Defaults to [].
    """
    doc = App.newDocument("MatrixSquares")
    rows, cols = len(matrix), len(matrix[0])

    # Select asymmetric patterns if mode is 1
    if mode == 1:
        candidates = list(set(asymmetric) & set(record))
        choice_patterns = random.sample(candidates, random.randint(1, len(candidates)))

    # Generate squares and place images
    for row in range(rows):
        for col in range(cols):
            x = col * unit_size
            y = (rows - 1 - row) * unit_size
            square = Part.makePlane(unit_size, unit_size, App.Vector(x, y, 0))
            obj = doc.addObject("Part::Feature", f"Square_{row}_{col}")
            obj.Shape = square

            if matrix[row][col] == 1:
                # In mode 1, pop from the record to ensure the same patterns are used
                if mode == 1:
                    image_name = record.pop(0)
                # In default mode, randomly select a pattern and record it
                else:
                    image_name = str(random.randint(1, 6))
                    record.append(image_name)

                plane = doc.addObject("Image::ImagePlane", f"Image_{row}_{col}")
                if mode == 1:
                    # Use a transformed image if the pattern is in the asymmetric choices
                    if image_name in choice_patterns:
                        plane.ImageFile = image_paths[f"{image_name}_0"]
                    # Otherwise, use the original image from the record
                    else:
                        plane.ImageFile = image_paths[image_name]
                else:
                    plane.ImageFile = image_paths[image_name]

                plane.XSize = unit_size
                plane.YSize = unit_size
                plane.Placement.Base = App.Vector(x + unit_size / 2, y + unit_size / 2, 0)

    # Add corner marker
    FreeCADGui.runCommand('Std_DrawStyle', 4)
    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
    if mode == 1:
        # Pop the marker position from the record to match the original
        selected_corner = record.pop(0)
    else:
        selected_corner = random.choice(corners)
        record.append(selected_corner)

    if selected_corner == "top_left":
        x = -marker_size
        y = rows * unit_size
    elif selected_corner == "top_right":
        x = cols * unit_size
        y = rows * unit_size
    elif selected_corner == "bottom_left":
        x = -marker_size
        y = -marker_size
    elif selected_corner == "bottom_right":
        x = cols * unit_size
        y = -marker_size

    # Create and style the red corner marker
    red_square = Part.makePlane(unit_size // 4, unit_size // 4, App.Vector(x, y, 0))
    red_obj = doc.addObject("Part::Feature", "Corner_Marker")
    red_obj.Shape = red_square
    red_obj.ViewObject.ShapeColor = (1.0, 0.0, 0.0)
    red_obj.ViewObject.DiffuseColor = (1.0, 0.0, 0.0)
    red_obj.ViewObject.LineColor = (0.0, 0.0, 0.0)
    red_obj.ViewObject.DisplayMode = "Flat Lines"
    red_obj.ViewObject.Transparency = 0
    red_obj.ViewObject.update()

    # Save image
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewTop()
    FreeCADGui.updateGui()
    view.fitAll()
    if mode == 1:
        image_path = f"{save_dir}/{save_name}_{selected_corner}_incorrect_transform.png"
    else:
        image_path = f"{save_dir}/{save_name}_{selected_corner}.png"
    view.saveImage(image_path, 1280, 1024)


def main(args):
    """
    Generates a specified number of data samples for a 2D rotation benchmark
    with image patterns.
    
    This function iterates `num_samples` times, creating a random binary matrix
    in each loop. It then calls `create_squares` to generate a base image.
    If the generated pattern contains any asymmetric components, it calls
    `create_squares` again in `mode=1` to generate a corresponding transformed
    image for the same pattern.
    
    Args:
        args (argparse.Namespace): An object containing command-line arguments.
    """
    # You should prepare the images in advance
    # Define image paths for patterns
    # _0 suffix indicates a rotated version of the base image
    image_paths = {
        "1": os.path.join(args.image_folder, "1.png"),
        "2": os.path.join(args.image_folder, "2.png"),
        "2_0": os.path.join(args.image_folder, "2_0.png"),
        "3": os.path.join(args.image_folder, "3.png"),
        "3_0": os.path.join(args.image_folder, "3_0.png"),
        "4": os.path.join(args.image_folder, "4.png"),
        "5": os.path.join(args.image_folder, "5.png"),
        "6": os.path.join(args.image_folder, "6.png"),
        "6_0": os.path.join(args.image_folder, "6_0.png"),
    }

    x, y = args.grid_size

    for i in range(args.num_samples):
        # Generate a random binary matrix
        matrix = [[random.randint(0, 1) for _ in range(x)] for _ in range(y)]

        save_dir = os.path.join(args.save_folder, f"{i}-{x}-{y}")
        os.makedirs(save_dir, exist_ok=True)

        # Record the sequence of images and the marker for reproducibility
        record = []
        create_squares(matrix, save_dir, image_paths, save_name=f"{i}-{x}-{y}", record=record)

        # If asymmetric patterns exist, generate a transformed version
        if set(record) & set(['2', '3', '6']):
            create_squares(matrix, save_dir, image_paths, save_name=f"{i}-{x}-{y}", mode=1, asymmetric=['2', '3', '6'], record=record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D rotation benchmark data with FreeCAD and image patterns.")
    parser.add_argument("--save_folder", type=str, required=True,
                        help="Path to the folder where generated data will be saved, e.g., /path/to/save/folder")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to the folder containing input pattern images, e.g., /path/to/images")
    parser.add_argument("--grid_size", type=int, default=[3, 2], nargs=2, metavar=("X", "Y"),
                        help="Size of the grid (default: 3x3)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of data samples to generate")
    args = parser.parse_args()

    main(args)



"""
    # run in FreeCAD
    import sys
    sys.argv = [
        "create_2D_rotation_complex.py",
        "--save_folder", "/path/to/save/MentalRotation/2DRotation/Level1",
        "--image_folder", "/path/to/pics/patterns",
        "--grid_size", "3 2",
        "--num_samples", "50"
    ]
    exec(open("/path/to/create_2D_rotation_complex.py").read())
"""