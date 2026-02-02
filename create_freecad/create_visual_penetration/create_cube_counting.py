import FreeCAD as App
from FreeCAD import Units
import FreeCADGui, TechDrawGui
import Part
import random, time, os
from collections import deque
import argparse

def create_cube(x, y, z, placement_space):
    """
    Creates a solid cube at a specific position.

    Args:
        x (int): The x-coordinate of the cube.
        y (int): The y-coordinate of the cube.
        z (int): The z-coordinate of the cube.
        placement_space (list): A 3D list representing the grid to track cube placement.
    """
    cube = Part.makeBox(cube_size, cube_size, cube_size)
    obj = doc.addObject("Part::Feature", f"Cube_{x}_{y}_{z}")
    obj.Shape = cube
    if placement_space[z][y][x] == 0:
        obj.Placement.Base = App.Vector(x, y, z)
        placement_space[z][y][x] = 1
    global min_x, min_y, min_z, max_x, max_y, max_z
    min_x = min(min_x, x)
    min_y = min(min_y, y)
    min_z = min(min_z, z)
    max_x = max(max_x, x)
    max_y = max(max_y, y)
    max_z = max(max_z, z)

def connect_isolated_cubes(placement_space, space_size_x, space_size_y):
    """
    Connects isolated groups of cubes on the bottom layer to ensure a single, contiguous object.

    Args:
        placement_space (list): The 3D list tracking cube positions.
        space_size_x (int): The width of the grid space.
        space_size_y (int): The depth of the grid space.
    """
    cubes = [(x, y) for x in range(space_size_x) for y in range(space_size_y) if placement_space[0][y][x] == 1]
    
    if not cubes:
        return  

    visited = set()
    regions = []
    
    for x, y in cubes:
        if (x, y) not in visited:
            region = []
            queue = deque()
            queue.append((x, y))
            visited.add((x, y))
            while queue:
                cx, cy = queue.popleft()
                region.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < space_size_x and 0 <= ny < space_size_y:
                        if (nx, ny) not in visited and placement_space[0][ny][nx] == 1:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            regions.append(region)
    
    if len(regions) > 1: 
        for i in range(len(regions) - 1):
            region1 = regions[i]
            region2 = regions[i + 1]
           
            min_dist = float('inf')
            best_pair = None
            for x1, y1 in region1:
                for x2, y2 in region2:
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = ((x1, y1), (x2, y2))

            (x1, y1), (x2, y2) = best_pair
            dx = 1 if x2 > x1 else -1
            dy = 1 if y2 > y1 else -1
            x, y = x1, y1
            while x != x2 or y != y2:
                if x != x2 and y != y2:
                    dx = 1 if x2 > x1 else -1
                    dy = 1 if y2 > y1 else -1
                    x += dx
                    y += dy
                elif x != x2:
                    dx = 1 if x2 > x1 else -1
                    x += dx
                elif y != y2:
                    dy = 1 if y2 > y1 else -1
                    y += dy
                if placement_space[0][y][x] == 0:
                    create_cube(x, y, 0, placement_space)

def main(args):
    """
    Main function to generate 3D cube counting puzzles and associated images and PDFs.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    num_samples = args.num_samples
    save_folder = args.save_folder
    template_path = args.template_path
    space_size_x, space_size_y, space_size_z = args.space_size
    global cube_size
    cube_size = 1
    
    for i in range(num_samples):
        global min_x, min_y, min_z, max_x, max_y, max_z
        placement_space = [[[0 for _ in range(space_size_x)] for _ in range(space_size_y)] for _ in range(space_size_z)]
        min_x = space_size_x
        min_y = space_size_y
        min_z = space_size_z
        max_x = 0
        max_y = 0
        max_z = 0

        doc = App.newDocument("Cubes")
        
        for z in range(space_size_z):
            for y in range(space_size_y):
                for x in range(space_size_x):
                    if z == 0 or (z > 0 and placement_space[z-1][y][x] == 1):
                        if random.choice([True, False]):
                            create_cube(x, y, z, placement_space)

        connect_isolated_cubes(placement_space, space_size_x, space_size_y)
            
        actual_space_size_x = max_x - min_x + 1
        actual_space_size_y = max_y - min_y + 1
        actual_space_size_z = max_z - min_z + 1
        if actual_space_size_x < 3 or actual_space_size_y < 3 or actual_space_size_z < 3:
            App.closeDocument(doc.Name)
            continue
        
        doc.recompute()

        compound = Part.makeCompound([obj.Shape for obj in doc.Objects if "Cube" in obj.Name])
        compound_obj = doc.addObject("Part::Feature", "Compound")
        compound_obj.Shape = compound
        solids_count = len(compound.Solids)
        
        save_dir = os.path.join(save_folder, f"{i}-{actual_space_size_x}-{actual_space_size_y}-{actual_space_size_z}-{solids_count}")
        os.makedirs(save_dir, exist_ok=True)

        bbx = compound_obj.Shape.BoundBox
        part_X = bbx.XLength
        part_Y = bbx.YLength
        part_Z = bbx.ZLength
            
        view = FreeCADGui.ActiveDocument.ActiveView
        view.viewIsometric()
        FreeCADGui.updateGui()

        page = doc.addObject('TechDraw::DrawPage', 'Drawing')
        template = doc.addObject('TechDraw::DrawSVGTemplate', 'Template')
        template.Template = template_path
        page.Template = template
        page_width= page.Template.Width
        page_height = page.Template.Height

        page.Visibility = False
        page.Visibility = True

        proj_group = doc.addObject('TechDraw::DrawProjGroup', 'ProjGroup')
        page.addView(proj_group)
        proj_group.Source = [compound_obj]

        proj_group.ProjectionType = "First Angle"
        proj_group.ScaleType = "Custom" 
        proj_group.Scale = min(float(page_width) / (float(part_X) + float(part_Y)), float(page_height) / (float(part_Z) + float(part_Y))) * 0.8 

        front_view = proj_group.addProjection("Front")
        left_view = proj_group.addProjection("Left")
        top_view = proj_group.addProjection("Top")

        front_view_width = Units.Quantity(part_X * proj_group.Scale, 'mm')
        front_view_height = Units.Quantity(part_Z * proj_group.Scale, 'mm')

        proj_group.X = front_view_width/2 + proj_group.spacingX*1.0
        proj_group.Y = page_height - (front_view_height/2 + proj_group.spacingY)

        doc.recompute()
        page.Visibility = True
        proj_group.Visibility = True

        FreeCADGui.updateGui()
        time.sleep(0.5)
        TechDrawGui.exportPageAsPdf(page, os.path.join(save_dir, f"{i}-{actual_space_size_x}-{actual_space_size_y}-{actual_space_size_z}-{solids_count}_3View.pdf"))

        view.fitAll()
        image_path = os.path.join(save_dir, f"{i}-{actual_space_size_x}-{actual_space_size_y}-{actual_space_size_z}-{solids_count}_Isometric.png")
        view.saveImage(image_path, 1280, 1024)
        
        # Part.export([compound_obj], os.path.join(save_dir, f"{i}-{actual_space_size_x}-{actual_space_size_y}-{actual_space_size_z}-{solids_count}.step"))
        
        App.closeDocument(doc.Name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate cube counting puzzles with isometric and multi-view projections.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to the directory to save the generated images and PDFs.")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the TechDraw SVG template file, e.g., /path/to/A4_Landscape_blank.svg")
    parser.add_argument("--space_size", type=int, default=[3, 3, 3], nargs=3, metavar=("X", "Y", "Z"),
                        help="Size of the space (default: 3x3x3)")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of data samples to generate.")
    
    args = parser.parse_args()
    main(args)

"""
    # run in FreeCAD
    import sys
    sys.argv = [
        "create_cube_counting.py",
        "--save_folder", "/path/to/save",
        "--template_path", "path/to/TechDraw SVG template"
        "--space_size", "3 3 3",
        "--num_samples", "50"
    ]
    exec(open("/path/to/create_cube_counting.py").read())
"""