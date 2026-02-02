import FreeCAD as App
from FreeCAD import Base
from pivy import coin
import Part
import TechDrawGui
import FreeCADGui
import random, math, time, os, itertools
import argparse

def create_cube(x, y, z):
    """
    Creates a solid cube object.
    
    Args:
        x (float): The length of the cube.
        y (float): The width of the cube.
        z (float): The height of the cube.
    """
    cube = Part.makeBox(x, y, z)
    cube_obj = doc.addObject("Part::Feature", "Obj_Cube")
    cube_obj.Shape = cube
    cube_obj.Placement.Base = App.Vector(-x/2, -y/2, 0)

def create_cylinder(r, h):
    """
    Creates a solid cylinder object.
    
    Args:
        r (float): The radius of the cylinder.
        h (float): The height of the cylinder.
    """
    cylinder = Part.makeCylinder(r, h)
    cylinder_obj = doc.addObject("Part::Feature", "Obj_Cylinder")
    cylinder_obj.Shape = cylinder
    cylinder_obj.Placement.Base = App.Vector(0, 0, 0)
    
def create_cone(r, h):
    """
    Creates a solid cone object.
    
    Args:
        r (float): The base radius of the cone.
        h (float): The height of the cone.
    """
    cone = Part.makeCone(r, 0, h)
    cone_obj = doc.addObject("Part::Feature", "Obj_Cone")
    cone_obj.Shape = cone
    cone_obj.Placement.Base = App.Vector(0, 0, 0)

def create_frustum_circle(r1, r2, h):
    """
    Creates a circular frustum (truncated cone).
    
    Args:
        r1 (float): The bottom radius.
        r2 (float): The top radius.
        h (float): The height of the frustum.
    """
    frustum = Part.makeCone(r1, r2, h)
    frustum_obj = doc.addObject("Part::Feature", "Obj_Frustum_Circle")
    frustum_obj.Shape = frustum
    frustum_obj.Placement.Base = App.Vector(0, 0, 0)
    
def create_tube(ro, ri, h):
    """
    Creates a hollow tube object by cutting a smaller cylinder from a larger one.
    
    Args:
        ro (float): The outer radius.
        ri (float): The inner radius.
        h (float): The height of the tube.
    """    
    outer_cylinder = Part.makeCylinder(ro, h)
    inner_cylinder = Part.makeCylinder(ri, h)
    tube = outer_cylinder.cut(inner_cylinder)
    tube_obj = doc.addObject("Part::Feature", "Obj_Tube")
    tube_obj.Shape = tube
    tube_obj.Placement.Base = App.Vector(0, 0, 0)

def create_triangular_prism(c, h):
    """
    Creates a triangular prism with an equilateral triangular base.
    
    Args:
        c (float): The side length of the equilateral base triangle.
        h (float): The height of the prism.
    """
    triangular_prism_obj = doc.addObject("Part::Feature", "Obj_Triangular_Prism")
    
    base_triangle_height = (math.sqrt(3) / 2) * c

    v1 = App.Vector(-c / 2, -base_triangle_height / 3, 0)
    v2 = App.Vector(c / 2, -base_triangle_height / 3, 0)
    v3 = App.Vector(0, (2 * base_triangle_height) / 3, 0)

    v4 = App.Vector(-c / 2, -base_triangle_height / 3, h)
    v5 = App.Vector(c / 2, -base_triangle_height / 3, h)
    v6 = App.Vector(0, (2 * base_triangle_height) / 3, h)

    base_triangle = Part.makePolygon([v1, v2, v3, v1])
    top_triangle = Part.makePolygon([v4, v5, v6, v4])

    face1 = Part.makePolygon([v1, v2, v5, v4, v1])
    face2 = Part.makePolygon([v2, v3, v6, v5, v2])
    face3 = Part.makePolygon([v3, v1, v4, v6, v3])

    base_face = Part.Face(base_triangle)
    top_face = Part.Face(top_triangle)

    side1 = Part.Face(face1)
    side2 = Part.Face(face2)
    side3 = Part.Face(face3)

    prism = Part.Shell([base_face, top_face, side1, side2, side3])
    solid = Part.Solid(prism)
    triangular_prism_obj.Shape = solid

def create_pyramid_triangle(c, h):
    """
    Creates a triangular pyramid with an equilateral triangular base.
    
    Args:
        c (float): The side length of the equilateral base triangle.
        h (float): The height of the pyramid.
    """
    pyramid_obj = doc.addObject("Part::Feature", "Obj_Pyramid_Triangle")
    bottom_triangle_height = (math.sqrt(3) / 2) * c

    v1 = App.Vector(-c / 2, -bottom_triangle_height / 3, 0)
    v2 = App.Vector(c / 2, -bottom_triangle_height / 3, 0)
    v3 = App.Vector(0, (2 * bottom_triangle_height) / 3, 0)

    v4 = App.Vector(0, 0, h) 

    triangle = Part.makePolygon([v1, v2, v3, v1])

    face1 = Part.makePolygon([v1, v2, v4, v1])
    face2 = Part.makePolygon([v2, v3, v4, v2])
    face3 = Part.makePolygon([v3, v1, v4, v3])

    base = Part.Face(triangle)

    side1 = Part.Face(face1)
    side2 = Part.Face(face2)
    side3 = Part.Face(face3)

    pyramid = Part.Shell([base, side1, side2, side3])
    solid = Part.Solid(pyramid)
    pyramid_obj.Shape = solid
    
def create_frustum_triangle(c1, c2, h):
    """
    Creates a triangular frustum (truncated triangular pyramid).
    
    Args:
        c1 (float): The side length of the bottom equilateral triangle.
        c2 (float): The side length of the top equilateral triangle.
        h (float): The height of the frustum.
    """
    frustum_obj = doc.addObject("Part::Feature", "Obj_Frustum_Triangle")
    
    base_triangle_height = (math.sqrt(3) / 2) * c1
    top_triangle_height = (math.sqrt(3) / 2) * c2

    v1 = App.Vector(-c1 / 2, -base_triangle_height / 3, 0)
    v2 = App.Vector(c1 / 2, -base_triangle_height / 3, 0)
    v3 = App.Vector(0, (2 * base_triangle_height) / 3, 0)

    v4 = App.Vector(-c2 / 2, -top_triangle_height / 3, h)
    v5 = App.Vector(c2 / 2, -top_triangle_height / 3, h)
    v6 = App.Vector(0, (2 * top_triangle_height) / 3, h)

    base_triangle = Part.makePolygon([v1, v2, v3, v1])
    top_triangle = Part.makePolygon([v4, v5, v6, v4])

    face1 = Part.makePolygon([v1, v2, v5, v4, v1])
    face2 = Part.makePolygon([v2, v3, v6, v5, v2])
    face3 = Part.makePolygon([v3, v1, v4, v6, v3])

    base_face = Part.Face(base_triangle)
    top_face = Part.Face(top_triangle)

    side1 = Part.Face(face1)
    side2 = Part.Face(face2)
    side3 = Part.Face(face3)

    frustum = Part.Shell([base_face, top_face, side1, side2, side3])
    solid = Part.Solid(frustum)
    frustum_obj.Shape = solid

def create_pyramid_square(c, h):
    """
    Creates a square pyramid.
    
    Args:
        c (float): The side length of the square base.
        h (float): The height of the pyramid.
    """
    pyramid_obj = doc.addObject("Part::Feature", "Obj_Pyramid_Square")
    half_side = c / 2

    v1 = App.Vector(-half_side, -half_side, 0)
    v2 = App.Vector(half_side, -half_side, 0)
    v3 = App.Vector(half_side, half_side, 0)
    v4 = App.Vector(-half_side, half_side, 0)

    v5 = App.Vector(0, 0, h) 

    square = Part.makePolygon([v1, v2, v3, v4, v1])

    face1 = Part.makePolygon([v1, v2, v5, v1])
    face2 = Part.makePolygon([v2, v3, v5, v2])
    face3 = Part.makePolygon([v3, v4, v5, v3])
    face4 = Part.makePolygon([v4, v1, v5, v4])

    base = Part.Face(square)

    side1 = Part.Face(face1)
    side2 = Part.Face(face2)
    side3 = Part.Face(face3)
    side4 = Part.Face(face4)

    pyramid = Part.Shell([base, side1, side2, side3, side4])
    solid = Part.Solid(pyramid)
    pyramid_obj.Shape = solid

def create_frustum_square(c1, c2, h):
    """
    Creates a square frustum (truncated square pyramid).
    
    Args:
        c1 (float): The side length of the bottom square.
        c2 (float): The side length of the top square.
        h (float): The height of the frustum.
    """
    frustum_obj = doc.addObject("Part::Feature", "Obj_Frustum_Square")
    
    base_half_side = c1 / 2
    top_half_side = c2 / 2

    v1 = App.Vector(-base_half_side, -base_half_side, 0)
    v2 = App.Vector(base_half_side, -base_half_side, 0)
    v3 = App.Vector(base_half_side, base_half_side, 0)
    v4 = App.Vector(-base_half_side, base_half_side, 0)

    v5 = App.Vector(-top_half_side, -top_half_side, h)
    v6 = App.Vector(top_half_side, -top_half_side, h)
    v7 = App.Vector(top_half_side, top_half_side, h)
    v8 = App.Vector(-top_half_side, top_half_side, h)

    base_square = Part.makePolygon([v1, v2, v3, v4, v1])
    top_square = Part.makePolygon([v5, v6, v7, v8, v5])

    face1 = Part.makePolygon([v1, v2, v6, v5, v1])
    face2 = Part.makePolygon([v2, v3, v7, v6, v2])
    face3 = Part.makePolygon([v3, v4, v8, v7, v3])
    face4 = Part.makePolygon([v4, v1, v5, v8, v4])

    base_face = Part.Face(base_square)
    top_face = Part.Face(top_square)

    side1 = Part.Face(face1)
    side2 = Part.Face(face2)
    side3 = Part.Face(face3)
    side4 = Part.Face(face4)

    frustum = Part.Shell([base_face, top_face, side1, side2, side3, side4])
    solid = Part.Solid(frustum)

    frustum_obj.Shape = solid

def create_objects_in_order(objects):
    """
    Creates a stack of 3D objects in the specified order and returns the compound shape.
    
    Args:
        objects (list): A list of strings representing the types of objects to create.
        
    Returns:
        tuple: A tuple containing the final compound shape and the FreeCAD object.
    """
    previous_obj = None
    compound_list = []
    
    for i, obj in enumerate(reversed(objects)):
        c = [10-2*i, 12-2*i]
        random.shuffle(c)
        h = random.choice([4, 6, 8])
        
        if obj == "cube":
            create_cube(c[0], c[0], random.choice([c[0], h]))
            current_obj = doc.getObject("Obj_Cube")
        elif obj == "cylinder":
            create_cylinder(c[0]/2, h)
            current_obj = doc.getObject("Obj_Cylinder")
        elif obj == "cone":
            create_cone(c[0]/2, h)
            current_obj = doc.getObject("Obj_Cone")
        elif obj == "frustum_circle":
            create_frustum_circle(c[0]/2, c[1]/2, h)
            current_obj = doc.getObject("Obj_Frustum_Circle")
        elif obj == "triangular_prism":
            create_triangular_prism(c[0], h)
            current_obj = doc.getObject("Obj_Triangular_Prism")
        elif obj == "pyramid_triangle":
            create_pyramid_triangle(c[0], h)
            current_obj = doc.getObject("Obj_Pyramid_Triangle")
        elif obj == "frustum_triangle":
            create_frustum_triangle(c[0], c[1], h)
            current_obj = doc.getObject("Obj_Frustum_Triangle")
        elif obj == "pyramid_square":
            create_pyramid_square(c[0], h)
            current_obj = doc.getObject("Obj_Pyramid_Square")
        elif obj == "frustum_square":
            create_frustum_square(c[0], c[1], h)
            current_obj = doc.getObject("Obj_Frustum_Square")
        
        if previous_obj is not None:
            previous_top = previous_obj.Shape.BoundBox.ZMax
            current_obj.Placement.Base.z = previous_top
        
        compound_list.append(current_obj.Shape)

        previous_obj = current_obj
    
    compound = Part.makeCompound(compound_list)
    compound_obj = doc.addObject("Part::Feature", "Compound")
    compound_obj.Shape = compound
        
    return compound, compound_obj


def get_sections_parallel_to_YZ(compound, compound_obj, k=3, mode=0, save_dir=None):
    """
    Generates cross-sections parallel to the YZ plane and saves images.
    
    Args:
        compound (Part.Compound): The compound shape to slice.
        compound_obj (FreeCAD object): The FreeCAD object containing the compound shape.
        k (int): The number of sections to generate.
        mode (int): 0 for standard sections, 1 for an "incorrect" section.
        save_dir (str): The directory to save the images.
    """
    x_min = compound_obj.Shape.BoundBox.XMin
    x_max = compound_obj.Shape.BoundBox.XMax
    x_step = (x_max - x_min) / (k+1)
    i_ = (k + 1)//2

    time.sleep(1)
    section_objs = []
    for i in range(1, k+1):  
        x_offset = x_min + x_step * i 
        direction = App.Vector(1, 0, 0) 
        sections = compound.slice(direction, x_offset) 
        section = Part.Compound(sections)  
        section_obj = doc.addObject("Part::Feature", f"Section_Parallel_YZ_{i}")
        section_obj.Shape = section
        section_objs.append(section_obj)
    
    
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewRight()
    FreeCADGui.updateGui()
    view.fitAll()
    time.sleep(1)
    
    for section_obj in section_objs:
        section_obj.ViewObject.Visibility = False
    
    for i, section_obj in enumerate(section_objs):
        section_obj.ViewObject.Visibility = True
        time.sleep(1)
        
        if mode == 0:
            image_path = os.path.join(save_dir, f"Section_Parallel_YZ_{i+1}.png")
            view.saveImage(image_path, 1280, 1024)
        if mode == 1 and i_ == (i+1):
            image_path = os.path.join(save_dir, f"incorrect_Section_Parallel_YZ_{i+1}.png")
            view.saveImage(image_path, 1280, 1024)
        
        section_obj.ViewObject.Visibility = False



def get_sections_parallel_to_XZ(compound, compound_obj, k=3, mode=0, save_dir=None):
    """
    Generates cross-sections parallel to the XZ plane and saves images.
    
    Args:
        compound (Part.Compound): The compound shape to slice.
        compound_obj (FreeCAD object): The FreeCAD object containing the compound shape.
        k (int): The number of sections to generate.
        mode (int): 0 for standard sections, 1 for an "incorrect" section.
        save_dir (str): The directory to save the images.
    """
    y_min = compound_obj.Shape.BoundBox.YMin
    y_max = compound_obj.Shape.BoundBox.YMax
    y_step = (y_max - y_min) / (k+1)
    i_ = (k + 1)//2
    
    section_objs = []
    time.sleep(1)
    for i in range(1, k+1):  
        y_offset = y_min + y_step * i 
        direction = App.Vector(0, 1, 0) 
        sections = compound.slice(direction, y_offset) 
        section = Part.Compound(sections)  
        section_obj = doc.addObject("Part::Feature", f"Section_Parallel_XZ_{i}")
        section_obj.Shape = section
        section_objs.append(section_obj)
        
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewFront()
    FreeCADGui.updateGui()
    view.fitAll()
    time.sleep(1)
    
    for section_obj in section_objs:
        section_obj.ViewObject.Visibility = False
    
    for i, section_obj in enumerate(section_objs):
        section_obj.ViewObject.Visibility = True 
        time.sleep(1)   
        
        if mode == 0:
            image_path = os.path.join(save_dir, f"Section_Parallel_XZ_{i+1}.png")
            view.saveImage(image_path, 1280, 1024)
        if mode == 1 and i_ == (i+1):
            image_path = os.path.join(save_dir, f"incorrect_Section_Parallel_XZ_{i+1}.png")
            view.saveImage(image_path, 1280, 1024)
        
        section_obj.ViewObject.Visibility = False


def get_sections_parallel_to_XY(compound, compound_obj, k=3, mode=0, save_dir=None):
    """
    Generates cross-sections parallel to the XY plane and saves images.
    
    Args:
        compound (Part.Compound): The compound shape to slice.
        compound_obj (FreeCAD object): The FreeCAD object containing the compound shape.
        k (int): The number of sections to generate.
        mode (int): 0 for standard sections, 1 for an "incorrect" section.
        save_dir (str): The directory to save the images.
    """
    z_min = compound_obj.Shape.BoundBox.ZMin
    z_max = compound_obj.Shape.BoundBox.ZMax
    z_step = (z_max - z_min) / (k+1)  
    i_ = (k + 1)//2
    
    section_objs = []
    time.sleep(1)
    for i in range(1, k+1):  
        z_offset = z_min + z_step * i  
        direction = App.Vector(0, 0, 1)  
        sections = compound.slice(direction, z_offset)  
        section = Part.Compound(sections)  
        section_obj = doc.addObject("Part::Feature", f"Section_Parallel_XY_{i}")
        section_obj.Shape = section
        section_objs.append(section_obj)
    
    
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewTop()
    FreeCADGui.updateGui()
    view.fitAll()
    time.sleep(1)
    
    for section_obj in section_objs:
        section_obj.ViewObject.Visibility = False
        
    for i, section_obj in enumerate(section_objs):
        section_obj.ViewObject.Visibility = True 
        time.sleep(1)  
           
        if mode == 0:
            image_path = os.path.join(save_dir, f"Section_Parallel_XY_{i+1}.png")
            view.saveImage(image_path, 1280, 1024)
        if mode == 1 and i_ == (i+1):
            image_path = os.path.join(save_dir, f"incorrect_Section_Parallel_XY_{i+1}.png")
            view.saveImage(image_path, 1280, 1024)
        
        section_obj.ViewObject.Visibility = False

def get_sections_rotated(compound, compound_center, mode=0, save_dir=None):
    """
    Generates rotated cross-sections and saves images.
    
    Args:
        compound (Part.Compound): The compound shape to slice.
        compound_center (App.Vector): The center point for slicing.
        mode (int): 0 for standard sections, 1 for "incorrect" sections.
        save_dir (str): The directory to save the images.
    """
    angles = [45, 135]
    axis_vector = {"x": App.Vector(1, 0, 0), "y": App.Vector(0, 1, 0), "z": App.Vector(0, 0, 1)}
    axis = random.choice(list(axis_vector.keys()))

    section_objs = []
    time.sleep(1)
    for angle in angles:
        radians = math.radians(angle)
        if axis == "x":
            direction = App.Vector(0, math.cos(radians), math.sin(radians))
        elif axis == "y":
            direction = App.Vector(math.cos(radians), 0, math.sin(radians))
        elif axis == "z":
            direction = App.Vector(math.cos(radians), math.sin(radians), 0)
            
        offset = direction.dot(compound_center)

        sections = compound.slice(direction, offset)
        section = Part.Compound(sections)

        section_obj = doc.addObject("Part::Feature", f"Section_Rotated_{angle}_{axis}")
        section_obj.Shape = section
        
        if axis == "x":
            section.rotate(compound_center, axis_vector[axis], angle+180)
        elif axis == "y":
            section.rotate(compound_center, axis_vector[axis], angle if angle==45 else angle+180)
        elif axis == "z":
            section.rotate(compound_center, axis_vector[axis], angle)

        section_obj.Shape = section
        section_objs.append(section_obj)
        
        
    view = FreeCADGui.ActiveDocument.ActiveView
    if axis == "x":
        view.viewTop()
    elif axis == "y":
        view.viewLeft()
    elif axis == "z":
        view.viewFront()
        
    FreeCADGui.updateGui()
    view.fitAll()
    time.sleep(1)
    
    for section_obj in section_objs:
        section_obj.ViewObject.Visibility = False
    
    for i, section_obj in enumerate(section_objs):
        section_obj.ViewObject.Visibility = True  
        time.sleep(1)
        
        if mode == 0:
            image_path = os.path.join(save_dir, f"Section_Rotated_{angles[i]}_{axis}.png")
            view.saveImage(image_path, 1280, 1024)
        if mode == 1:
            image_path = os.path.join(save_dir, f"incorrect_Section_Rotated_{angles[i]}_{axis}.png")
            view.saveImage(image_path, 1280, 1024)
        
        section_obj.ViewObject.Visibility = False

def adjust_object_sizes(compound_obj):
    """
    Adjusts the sizes of the individual shapes within a compound object and returns a new compound shape.
    
    Args:
        compound_obj (FreeCAD object): The FreeCAD object containing the compound shape.
    
    Returns:
        tuple: A tuple containing the new compound shape and its FreeCAD object.
    """
    shapes = list(compound_obj.Shape.childShapes())
        
    adjusted_shapes = []
    original_bottom = shapes[0].BoundBox.ZMin

    base_scale_factors = [0.5, 1.0, 2.0]
    height_scale_factors = [0.5, 1.0, 2.0]
    all_scale_combinations = list(itertools.product(base_scale_factors, height_scale_factors))
    all_scale_combinations.remove((0.5, 0.5))
    all_scale_combinations.remove((1.0, 1.0))
    all_scale_combinations.remove((2.0, 2.0))
    random.shuffle(all_scale_combinations)

    for i, shape in enumerate(shapes): 
        base_scale_factor, height_scale_factor = all_scale_combinations[i]
        
        scaled_shape = shape.copy()
        
        matrix = Base.Matrix()
        matrix.scale(base_scale_factor, base_scale_factor, height_scale_factor)
        scaled_shape.transformShape(matrix)
        
        bbox = scaled_shape.BoundBox
    
        if i == 0:
            translation = Base.Vector(0, 0, original_bottom - bbox.ZMin)
            scaled_shape.Placement.move(translation)
        else:
            prev_shape = adjusted_shapes[i - 1]
            prev_top = prev_shape.BoundBox.ZMax
            current_bottom = bbox.ZMin
            translation = Base.Vector(0, 0, prev_top - current_bottom)
            scaled_shape.Placement.move(translation)
            
        adjusted_shapes.append(scaled_shape)

    new_compound = Part.makeCompound(adjusted_shapes)

    new_compound_obj = doc.addObject("Part::Feature", "Adjusted_Compound")
    new_compound_obj.Shape = new_compound
    return new_compound, new_compound_obj

def main(args):
    """
    Main function to generate 3D models and cross-section images based on provided arguments.
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    num_samples = args.num_samples
    save_folder = args.save_folder
    
    num_objs = args.num_objs
    for i in range(num_samples):
        doc = App.newDocument()

        objects_list = ["cube",  "cylinder", "frustum_circle", "triangular_prism", "frustum_triangle", "frustum_square"]
        pyramid_object_list = ["cone", "pyramid_triangle", "pyramid_square"]

        selected_objects = random.sample(objects_list, random.randint(num_objs-1, num_objs))
        if len(selected_objects) == num_objs:
            selected_pyramid = []
        else:
            selected_pyramid = random.sample(pyramid_object_list, 1)
        sorted_objects = selected_pyramid + selected_objects

        save_dir = os.path.join(save_folder, f"{i}-{'-'.join(sorted_objects)}")
        os.makedirs(save_dir, exist_ok=True)

        compound, compound_obj = create_objects_in_order(sorted_objects)

        for o in [obj for obj in doc.Objects if "Obj" in obj.Name]:
            o.ViewObject.Visibility = False

        view = FreeCADGui.ActiveDocument.ActiveView
        view.viewAxonometric()
        FreeCADGui.updateGui()
        view.fitAll()
        time.sleep(1)
        image_path = os.path.join(save_dir, "Isometric.png")
        view.saveImage(image_path, 1280, 1024)
        time.sleep(1)
        
        view = FreeCADGui.ActiveDocument.ActiveView
        cam = view.getCameraNode()
        rot = coin.SbRotation(coin.SbVec3f(0, 0, 1), 3.14159 * 0.7) 
        current_orientation = cam.orientation.getValue()
        new_orientation = current_orientation * rot
        cam.orientation.setValue(new_orientation)
        FreeCADGui.updateGui()
        view.fitAll()
        time.sleep(1)
        image_path = os.path.join(save_dir, "Isometric_rotate.png")
        view.saveImage(image_path, 1280, 1024)
        view.fitAll()

        compound_obj.ViewObject.Visibility = False

        get_sections_parallel_to_YZ(compound, compound_obj, k=3, mode=0, save_dir=save_dir)
        get_sections_parallel_to_XY(compound, compound_obj, k=3, mode=0, save_dir=save_dir)
        get_sections_parallel_to_XZ(compound, compound_obj, k=3, mode=0, save_dir=save_dir)
        compound_center = compound_obj.Shape.BoundBox.Center
        get_sections_rotated(compound, compound_center, save_dir=save_dir)

        time.sleep(1)
        new_compound, new_compound_obj = adjust_object_sizes(compound_obj)
        new_compound_obj.ViewObject.Visibility = False
        choice = random.choice(["parallel_to_YZ", "parallel_to_XZ"])
        if choice == "parallel_to_YZ":
            get_sections_parallel_to_YZ(new_compound, new_compound_obj, k=3, mode=1, save_dir=save_dir)
        elif choice == "parallel_to_XZ":
            get_sections_parallel_to_XZ(new_compound, new_compound_obj, k=3, mode=1, save_dir=save_dir)
        elif choice == "rotated_around":
            get_sections_rotated(compound, compound_center, k=1, mode=1, save_dir=save_dir)
        
        doc.recompute()
        App.closeDocument(doc.Name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate 3D models and cross-section images for a spatial thinking benchmark.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to the directory to save the generated images.")
    parser.add_argument("--num_objs", type=int, default=2, help="Number of objects to compound.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of data samples to generate.")
    
    args = parser.parse_args()
    main(args)
    
"""
    # run in FreeCAD
    import sys

    sys.argv = [
        "create_cross_section.py",
        "--save_folder", "/path/to/save",
        "--o b j s", "2",
        "--num_samples", "50"
    ]

    exec(open("/path/to/create_cross_section.py").read())

"""