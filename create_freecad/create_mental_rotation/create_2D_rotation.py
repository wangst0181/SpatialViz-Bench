import os
import random
import argparse
import Part
import FreeCADGui
import time

# Define color mapping
color_value = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "pink": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0)
}

def create_squares(matrix, save_dir, unit_size=10, marker_size=2, save_name=None):
    """
    Create a grid of squares in FreeCAD from a binary matrix.
    
    This function visualizes a 2D binary matrix by creating FreeCAD square objects.
    Cells with a value of 1 are filled with a random color from a predefined set,
    while cells with a value of 0 are not filled but still have a visible outline.
    A red marker is placed at a random corner to provide an orientation reference.
    The final arrangement is saved as a PNG image.

    Args:
        matrix (list of list): A 2D binary matrix representing the grid.
        save_dir (str): The directory where the output image will be saved.
        unit_size (int, optional): The side length of each square. Defaults to 10.
        marker_size (int, optional): The side length of the corner marker. Defaults to 2.
        save_name (str, optional): The base name for the output image file. Defaults to None.
    """
    doc = App.newDocument("MatrixSquares")
    rows, cols = len(matrix), len(matrix[0])
    
    # Generate squares based on matrix values
    for row in range(rows):
        for col in range(cols):
            x = col * unit_size
            y = (rows - 1 - row) * unit_size  
            square = Part.makePlane(unit_size, unit_size, App.Vector(x, y, 0))
            obj = doc.addObject("Part::Feature", f"Square_{row}_{col}")
            obj.Shape = square
                
            if matrix[row][col] == 1:
                # Assign a random color to filled squares
                color = random.choice(list(color_value.values()))
                obj.ViewObject.ShapeColor = color
                obj.ViewObject.DiffuseColor = [color]  
                obj.ViewObject.LineColor = (0.0, 0.0, 0.0)
                obj.ViewObject.DisplayMode = "Flat Lines"
                obj.ViewObject.Transparency = 0 
                obj.ViewObject.update()
    
    # Randomly select one corner for marker placement
    FreeCADGui.runCommand('Std_DrawStyle', 4)
    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
    selected_corner = random.choice(corners)
    
    # Calculate marker coordinates
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
        
    # Add red corner marker
    red_square = Part.makePlane(unit_size // 4, unit_size // 4, App.Vector(x, y, 0))
    red_obj = doc.addObject("Part::Feature", "Corner_Marker")
    red_obj.Shape = red_square
    red_obj.ViewObject.ShapeColor = color_value["red"] 
    red_obj.ViewObject.DiffuseColor = [color_value["red"]]
    red_obj.ViewObject.LineColor = (0.0, 0.0, 0.0)
    red_obj.ViewObject.DisplayMode = "Flat Lines"
    red_obj.ViewObject.Transparency = 0
    red_obj.ViewObject.update()
    
    # Adjust view and save image
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewTop()
    FreeCADGui.updateGui()
    view.fitAll()
    image_path = os.path.join(save_dir, f"{save_name}_{selected_corner}.png")
    view.saveImage(image_path, 1280, 1024)

def main(args):
    """
    Generate a specified number of data samples for a 2D rotation benchmark.
    
    This function iterates a set number of times, generating a new random
    binary matrix in each loop. It then calls `create_squares` to visualize
    the matrix and save the resulting image.

    Args:
        args (argparse.Namespace): An object containing command-line arguments.
    """
    x, y = args.grid_size
    
    for i in range(args.num_samples):
        # Generate a random binary matrix
        matrix = [[random.randint(0, 1) for _ in range(x)] for _ in range(y)]
        
        save_dir = os.path.join(args.save_folder, f"{i}-{x}-{y}")
        os.makedirs(save_dir, exist_ok=True) 
        
        # Create squares and save the image
        create_squares(matrix, save_dir, save_name=f"{i}-{x}-{y}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D rotation benchmark data with FreeCAD.")
    parser.add_argument("--save_folder", type=str, required=True,
                        help="Path to the folder where generated data will be saved, e.g., /path/to/save/folder")
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
        "create_2D_rotation.py",
        "--save_folder", "/path/to/save",
        "--grid_size", "3 2",
        "--num_samples", "50"
    ]
    exec(open("/path/to/create_2D_rotation.py").read())
"""