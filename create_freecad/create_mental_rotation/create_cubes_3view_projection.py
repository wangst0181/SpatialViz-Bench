import FreeCAD as App
from FreeCAD import Units
import FreeCADGui, TechDrawGui
import Part
import random, time, os, argparse
from collections import deque


def create_cube(x, y, z, placement_space, doc, bounds):
    """
    Creates a cube at the specified (x,y,z) coordinates if the space is empty,
    and updates the bounding box of the overall structure.

    Args:
        x (int): The x-coordinate of the cube.
        y (int): The y-coordinate of the cube.
        z (int): The z-coordinate of the cube.
        placement_space (list): A 3D list tracking cube placement.
        doc (FreeCAD.Document): The FreeCAD document object.
        bounds (dict): A dictionary to track the structure's bounding box.
    """
    cube_size = 1
    cube = Part.makeBox(cube_size, cube_size, cube_size)
    obj = doc.addObject("Part::Feature", f"Cube_{x}_{y}_{z}")
    obj.Shape = cube
    if placement_space[z][y][x] == 0:
        obj.Placement.Base = App.Vector(x, y, z)
        placement_space[z][y][x] = 1
    bounds["min_x"] = min(bounds["min_x"], x)
    bounds["min_y"] = min(bounds["min_y"], y)
    bounds["min_z"] = min(bounds["min_z"], z)
    bounds["max_x"] = max(bounds["max_x"], x)
    bounds["max_y"] = max(bounds["max_y"], y)
    bounds["max_z"] = max(bounds["max_z"], z)
    return obj.ViewObject.ShapeColor


def connect_isolated_cubes(placement_space, space_size_x, space_size_y, doc, bounds):
    """
    Connects isolated 2D clusters of cubes on the bottom layer to ensure a single,
    contiguous object is formed.

    Args:
        placement_space (list): A 3D list tracking cube placement.
        space_size_x (int): The width of the grid.
        space_size_y (int): The depth of the grid.
        doc (FreeCAD.Document): The FreeCAD document object.
        bounds (dict): A dictionary to track the structure's bounding box.
    """
    cubes = [(x, y) for x in range(space_size_x) for y in range(space_size_y) if placement_space[0][y][x] == 1]
    if not cubes:
        return
    visited = set()
    regions = []
    for x, y in cubes:
        if (x, y) not in visited:
            region = []
            queue = deque([(x, y)])
            visited.add((x, y))
            while queue:
                cx, cy = queue.popleft()
                region.append((cx, cy))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < space_size_x and 0 <= ny < space_size_y:
                        if (nx, ny) not in visited and placement_space[0][ny][nx] == 1:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            regions.append(region)
    if len(regions) > 1:
        for i in range(len(regions) - 1):
            r1, r2 = regions[i], regions[i+1]
            min_dist, best_pair = float("inf"), None
            for x1, y1 in r1:
                for x2, y2 in r2:
                    dist = abs(x1-x2) + abs(y1-y2)
                    if dist < min_dist:
                        min_dist, best_pair = dist, ((x1,y1),(x2,y2))
            (x1,y1),(x2,y2) = best_pair
            x, y = x1, y1
            while x != x2 or y != y2:
                if x != x2: x += 1 if x2 > x else -1
                if y != y2: y += 1 if y2 > y else -1
                if placement_space[0][y][x] == 0:
                    create_cube(x, y, 0, placement_space, doc, bounds)


def is_face_visible(placement_space, x, y, z, direction, size_z):
    """
    Checks if a specific face of a cube is visible from a given viewing direction.

    Args:
        placement_space (list): The 3D list of cube placements.
        x (int): The x-coordinate of the cube.
        y (int): The y-coordinate of the cube.
        z (int): The z-coordinate of the cube.
        direction (str): The viewing direction ("top", "front", or "left").
        size_z (int): The z-dimension of the grid.

    Returns:
        bool: True if the face is visible, False otherwise.
    """
    if direction == "top":
        return all(placement_space[k][y][x] == 0 for k in range(z+1, size_z))
    elif direction == "front":
        return all(placement_space[z][k][x] == 0 for k in range(0, y))
    elif direction == "left":
        return all(placement_space[z][y][k] == 0 for k in range(0, x))
    return False


def color_visible_faces(placement_space, space_size_x, space_size_y, space_size_z, k, doc, mode=0):
    """
    Randomly selects and colors k visible faces red.

    Args:
        placement_space (list): The 3D list of cube placements.
        space_size_x (int): The x-dimension of the grid.
        space_size_y (int): The y-dimension of the grid.
        space_size_z (int): The z-dimension of the grid.
        k (int): The number of faces to color.
        doc (FreeCAD.Document): The FreeCAD document object.
        mode (int): 0 to color original cubes, 1 to color copied cubes.
    """
    visible_faces = []
    for z in range(space_size_z):
        for y in range(space_size_y):
            for x in range(space_size_x):
                if placement_space[z][y][x] == 1:
                    cube_pos = (x,y,z)
                    for d in ["top","front","left"]:
                        if is_face_visible(placement_space, x, y, z, d, space_size_z):
                            visible_faces.append((cube_pos, d))
    if not visible_faces: return
    k = min(k, len(visible_faces))
    selected_faces = random.sample(visible_faces, k)
    for (x,y,z), _ in selected_faces:
        cube_name = f"Cube_{x}_{y}_{z}" if mode == 0 else f"Cube_{x}_{y}_{z}_Copy"
        cube_obj = doc.getObject(cube_name)
        if cube_obj:
            red = (1.0, 0.0, 0.0)
            cube_obj.ViewObject.DiffuseColor = [red]*6
            cube_obj.ViewObject.DisplayMode = "Flat Lines"
            cube_obj.ViewObject.Transparency = 0
            cube_obj.ViewObject.update()
            doc.recompute()


def copy_all_cubes(doc):
    """
    Copies all existing cubes in the document to create a duplicate set.

    Args:
        doc (FreeCAD.Document): The FreeCAD document object.

    Returns:
        int: The number of cubes that were copied.
    """
    cubes = [obj for obj in doc.Objects if "Cube" in obj.Name and "Copy" not in obj.Name]
    for cube in cubes:
        copy = doc.addObject("Part::Feature", f"{cube.Name}_Copy")
        copy.Shape = cube.Shape
        copy.Placement = cube.Placement
        if hasattr(cube.ViewObject, "ShapeColor"):
            copy.ViewObject.ShapeColor = cube.ViewObject.ShapeColor
    return len(cubes)


def save_views(prefix, mode=0):
    """
    Saves images of the current scene from different viewpoints.

    Args:
        prefix (str): The filename prefix for the saved images.
        mode (int): 0 for saving isometric, front, top, left, and right views.
                    1 for saving only left and right views.
    """
    FreeCADGui.runCommand('Std_DrawStyle', 4)
    view = FreeCADGui.ActiveDocument.ActiveView
    if mode == 0:
        for func, name in [(view.viewIsometric,"Isometric"),
                           (view.viewFront,"Front"),
                           (view.viewTop,"Top"),
                           (view.viewLeft,"Left"),
                           (view.viewRight,"Right")]:
            time.sleep(1)
            func()
            FreeCADGui.updateGui()
            view.fitAll()
            time.sleep(1)
            view.saveImage(f"{prefix}_{name}.png", 1280, 1024)
    elif mode == 1:
        for func, name in [(view.viewLeft,"Left"), (view.viewRight,"Right")]:
            time.sleep(1)
            func()
            FreeCADGui.updateGui()
            view.fitAll()
            time.sleep(1)
            view.saveImage(f"{prefix}_{name}.png", 1280, 1024)


def main(args):
    """
    The main function to generate a cube-based mental rotation dataset with two versions
    (a correct one and an incorrect one).

    Args:
        args (argparse.Namespace): The object containing command-line arguments.
    """
    for i in range(args.num_samples):
        space_x, space_y, space_z = args.space_size
        placement_space = [[[0 for _ in range(space_x)] for _ in range(space_y)] for _ in range(space_z)]
        bounds = {"min_x":space_x,"min_y":space_y,"min_z":space_z,
                  "max_x":0,"max_y":0,"max_z":0}

        doc = App.newDocument("Cubes")
        for z in range(space_z):
            for y in range(space_y):
                for x in range(space_x):
                    if z==0 or (z>0 and placement_space[z-1][y][x]==1):
                        if random.choice([True, False]):
                            create_cube(x,y,z,placement_space,doc,bounds)

        connect_isolated_cubes(placement_space, space_x, space_y, doc, bounds)
        solids_count = copy_all_cubes(doc)
        doc.recompute()

        actual_x = bounds["max_x"]-bounds["min_x"]+1
        actual_y = bounds["max_y"]-bounds["min_y"]+1
        actual_z = bounds["max_z"]-bounds["min_z"]+1
        if actual_x < 3 or actual_y < 3 or actual_z < 3: 
            continue

        save_dir = os.path.join(args.save_folder, f"{i}-{actual_x}-{actual_y}-{actual_z}-{solids_count}")
        os.makedirs(save_dir, exist_ok=True)

        # Correct version
        for cube in [o for o in doc.Objects if "Copy" in o.Name]: cube.ViewObject.Visibility=False
        color_visible_faces(placement_space, space_x, space_y, space_z, args.num_visible_faces, doc, mode=0)
        save_views(os.path.join(save_dir,"cubes"), mode=0)

        # Incorrect version
        for cube in [o for o in doc.Objects if "Cube" in o.Name]:
            cube.ViewObject.Visibility = "Copy" in cube.Name
        color_visible_faces(placement_space, space_x, space_y, space_z, args.num_visible_faces, doc, mode=1)
        save_views(os.path.join(save_dir,"cubes_incorrect"), mode=1)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate cube-based mental rotation dataset.")
    parser.add_argument("--save_folder", type=str, required=True, help="Output folder path.")
    parser.add_argument("--space_size", type=int, default=[3, 3, 3], nargs=3, metavar=("X", "Y", "Z"),
                        help="Size of the space (default: 3x3x3)")
    parser.add_argument("--num_visible_faces", type=int, default=3, help="Number of faces to be colored in red.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    args = parser.parse_args()
    main(args)

"""
    # run in FreeCAD
    import sys
    sys.argv = [
        "create_cubes_3view_projection.py",
        "--save_folder", "/path/to/save/Cubes3View",
        "--space_size", "3 3 3",
        "--num_visible_faces", "3",
        "--num_samples", "40"
    ]
    exec(open("/path/to/create_cubes_3view_projection.py").read())

"""