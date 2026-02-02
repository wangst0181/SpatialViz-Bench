import random, os, json, shutil, time
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import *


def detect_grid(image_path, row_num, col_num):
    """
    Detects a grid of squares in an image and returns it as a binary matrix.
    Args:
        image_path (str): The path to the image file.
        row_num (int): The number of rows in the grid.
        col_num (int): The number of columns in the grid.
    Returns:
        np.array: A NumPy array representing the grid, with 1 for a filled square and 0 for an empty one.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read, please check the path.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Crop the white border
    points = cv2.findNonZero(binary)
    if points is None:
        return np.zeros((0, 0), dtype=int)
    x, y, w, h = cv2.boundingRect(points)
    cropped = binary[y:y+h, x:x+w]
    
    contours, _ = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    grid = np.zeros((row_num, col_num), dtype=int)
    # The first contour is the outer border, so we skip it (contours[1:])
    for cnt in contours[1:]:
        x, y, w, h = cv2.boundingRect(cnt)
        row = y // h
        col = x // w
        if 0 <= row < row_num and 0 <= col < col_num:
            grid[row][col] = 1
    return grid


def get_cube_ans(front, left, top, mode=0):
    """
    Calculates the maximum and minimum number of cubes required based on 2 or 3 views.
    Args:
        front (np.array): The grid for the front view.
        left (np.array): The grid for the left view.
        top (np.array): The grid for the top view.
        mode (int, optional): 0 for 2-view calculation (Front, Top), 1 for 3-view. Defaults to 0.
    Returns:
        tuple: A tuple containing (max_number_of_cubes, min_number_of_cubes).
    """
    height, width = top.shape
    sum_front = np.sum(front, axis=0)
    sum_top = np.sum(top, axis=0)
    max_num_2view = np.dot(sum_front, sum_top)
    min_num_2view = np.sum((sum_top - 1) + sum_front)
    if mode == 0:
        return int(max_num_2view), int(min_num_2view)
    
    sum_left = np.sum(left, axis=0)
    ans = np.zeros_like(top)
    for row in range(height):
        for col in range(width):
            if top[row][col] == 1:
                ans[row][col] = min(sum_front[col], sum_left[row])
    max_num_3view = np.sum(ans)
    sum_top_T = np.sum(top, axis=1)
    min_num_3view = max(np.sum((sum_top_T - 1) + sum_left), min_num_2view)
    
    if mode == 1:
        return int(max_num_3view), int(min_num_3view)
    

def create_composite_image_CubeCounting(folder_path, output_path, spacing=70, padding=100, font_size=60, mode=0):
    """
    Creates a composite reference image for cube counting problems.

    Args:
        folder_path (str): The path to the folder containing the source images.
        output_path (str): The path to save the composite image.
        spacing (int, optional): The spacing between images. Defaults to 70.
        padding (int, optional): The padding around the canvas. Defaults to 100.
        font_size (int, optional): The font size for the labels. Defaults to 60.
        mode (int, optional): The display mode (0 for Front-Top, 1 for Front-Left-Top). Defaults to 0.

    Raises:
        FileNotFoundError: If a required image is not found in the folder.
    """
    images = {}
    filenames = ["Front", "Left", "Top"]
    for name in filenames:
        img_path = os.path.join(folder_path, f"{name}.png")
        if os.path.exists(img_path):
            images[name] = Image.open(img_path)
        else:
            raise FileNotFoundError(f"Image '{name}.png' not found in the folder.")
    
    # Get the width and height of all sub-images
    widths = {name: img.width for name, img in images.items()}
    heights = {name: img.height for name, img in images.items()}
    
    if mode==0: # front-top
        upper_width = widths["Front"] + widths["Top"] + spacing
        upper_height = max(heights["Front"], heights["Top"])
        
        total_width = upper_width + 2 * padding
        total_height = upper_height + 2 * padding + font_size
        
        # Create a white canvas
        canvas = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(canvas)
        
        # Load font (uses system default font)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        
        # Calculate the starting x position for the upper row
        upper_x = (total_width - upper_width) // 2
        upper_y = padding + font_size
        
        # Place the upper row images
        canvas.paste(images["Front"], (upper_x, upper_y))
        canvas.paste(images["Top"], (upper_x + widths["Front"] + spacing, upper_y))
        
        # Add text above each image and ensure the font is visible
        text_positions = [
            (upper_x + widths["Front"] // 2, upper_y - font_size - 10, "Front"),
            (upper_x + widths["Front"] + spacing + widths["Top"] // 2, upper_y - font_size - 10, "Top"),
        ]
        
        for x, y, text in text_positions:
            bbox = draw.textbbox((x, y), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x - text_width // 2, y), text, fill="black", font=font)
        
        # Save the output image
        canvas.save(output_path, "PNG")
        print(f"Composite image saved at: {output_path}")
    
    elif mode == 1: # Front-Left-Top
        upper_width = widths["Front"] + widths["Top"] + widths["Left"] + 2 * spacing
        upper_height = max(heights["Front"], heights["Top"], heights["Left"])
        
        total_width = upper_width + 2 * padding
        total_height = upper_height + 2 * padding + font_size
        
        # Create a white canvas
        canvas = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(canvas)
        
        # Load font (uses system default font)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        
        # Calculate the starting x position for the upper row
        upper_x = (total_width - upper_width) // 2
        upper_y = padding + font_size
        
        # Place the upper row images
        canvas.paste(images["Front"], (upper_x, upper_y))
        canvas.paste(images["Top"], (upper_x + widths["Front"] + spacing, upper_y))
        canvas.paste(images["Left"], (upper_x + widths["Front"] + widths["Top"] + 2 * spacing, upper_y))
        
        # Add text above each image and ensure the font is visible
        text_positions = [
            (upper_x + widths["Front"] // 2, upper_y - font_size - 10, "Front"),
            (upper_x + widths["Front"] + spacing + widths["Top"] // 2, upper_y - font_size - 10, "Top"),
            (upper_x + widths["Front"] + widths["Top"] + 2 * spacing + widths["Left"] // 2, upper_y - font_size - 10, "Left")
        ]
        
        for x, y, text in text_positions:
            bbox = draw.textbbox((x, y), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x - text_width // 2, y), text, fill="black", font=font)
        
        # Save the output image
        canvas.save(output_path, "PNG")
        print(f"Composite image saved at: {output_path}")
    
    
def create_composite_image_CubeAssembly(folder_path, output_path, spacing=70, padding=100, font_size=60):
    """
    Creates a composite reference image for sliding blocks problems.

    Args:
        folder_path (str): The path to the folder containing the source images.
        output_path (str): The path to save the composite image.
        spacing (int, optional): The spacing between images. Defaults to 70.
        padding (int, optional): The padding around the canvas. Defaults to 100.
        font_size (int, optional): The font size for the labels. Defaults to 60.

    Raises:
        FileNotFoundError: If a required image is not found in the folder.
    """
    # Read all PNG images
    images = {}
    filenames = {"total": "", "0": "", "1": "", "A": "", "B": "", "C": ""}
    for file in os.listdir(folder_path):
        if file.split('_')[-1].split('.')[0] in filenames:
            filenames[file.split('_')[-1].split('.')[0]] = file
 
    for key, value in filenames.items():
        if not value:
            continue
        img_path = os.path.join(folder_path, value)
        if os.path.exists(img_path):
            images[key] = Image.open(img_path)
        else:
            raise FileNotFoundError(f"Image {value} not found in the folder.")
    
    # Get the width and height of all sub-images
    widths = {name: img.width for name, img in images.items()}
    heights = {name: img.height for name, img in images.items()}
    
     # Calculate the total width of the upper and lower rows
    if not filenames["1"]:
        upper_width = widths["total"] + widths["0"] + spacing
        upper_height = max(heights["total"], heights["0"])
    else:
        upper_width = widths["total"] + widths["0"] + widths["1"] + 2 * spacing
        upper_height = max(heights["total"], heights["0"], heights["1"])
    
    lower_width = widths["A"] + widths["B"] + widths["C"] + 2 * spacing
    max_total_width = max(upper_width, lower_width)
    
    lower_height = max(heights["A"], heights["B"], heights["C"])
    
    # Calculate the total height of the canvas to ensure all sub-images are fully visible
    total_width = max_total_width + 2 * padding
    total_height = upper_height + lower_height + 3 * padding + 2 * font_size
    
    # Create a white canvas
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    
    # Load font (uses system default font)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    
    # Calculate the starting x position for the upper row
    upper_x = (total_width - upper_width) // 2
    upper_y = padding + font_size
    
    # Place the upper row images
    canvas.paste(images["total"], (upper_x, upper_y))
    canvas.paste(images["0"], (upper_x + widths["total"] + spacing, upper_y))
    if filenames["1"]:
        canvas.paste(images["1"], (upper_x + widths["total"] +  widths["0"] + 2 * spacing, upper_y))
    
    # Calculate the starting x position for the lower row
    lower_x = (total_width - lower_width) // 2
    lower_y = upper_y + upper_height + padding + font_size
    
    # Place the lower row images
    canvas.paste(images["A"], (lower_x, lower_y))
    canvas.paste(images["B"], (lower_x + widths["A"] + spacing, lower_y))
    canvas.paste(images["C"], (lower_x + widths["A"] + widths["B"] + 2 * spacing, lower_y))
    
    # Add text above each image and ensure the font is visible
    if filenames["1"]:
        text_positions = [
            (upper_x + widths["total"] // 2, upper_y - font_size - 10, "Complete Cube Stack"),
            (upper_x + widths["total"] + spacing + widths["0"] // 2, upper_y - font_size - 10, "Part 1"),
            (upper_x + widths["total"] + widths["0"] + 2 * spacing + widths["1"] // 2, upper_y - font_size - 10, "Part 2"),
            (lower_x + widths["A"] // 2, lower_y - font_size - 10, "A"),
            (lower_x + widths["A"] + spacing + widths["B"] // 2, lower_y - font_size - 10, "B"),
            (lower_x + widths["A"] + widths["B"] + 2 * spacing + widths["C"] // 2, lower_y - font_size - 10, "C")
        ]
    else:
        text_positions = [
            (upper_x + widths["total"] // 2, upper_y - font_size - 10, "Complete Cube Stack"),
            (upper_x + widths["total"] + spacing + widths["0"] // 2, upper_y - font_size - 10, "Part 1"),
            (lower_x + widths["A"] // 2, lower_y - font_size - 10, "A"),
            (lower_x + widths["A"] + spacing + widths["B"] // 2, lower_y - font_size - 10, "B"),
            (lower_x + widths["A"] + widths["B"] + 2 * spacing + widths["C"] // 2, lower_y - font_size - 10, "C")
        ]
    
    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")
    

def create_composite_image_CrossSection(folder_path, output_path, spacing=70, padding=100, font_size=60):
    """
    Creates a composite reference image for cross-section problems.

    Args:
        folder_path (str): The path to the folder containing the source images.
        output_path (str): The path to save the composite image.
        spacing (int, optional): The spacing between images. Defaults to 70.
        padding (int, optional): The padding around the canvas. Defaults to 100.
        font_size (int, optional): The font size for the labels. Defaults to 60.

    Raises:
        FileNotFoundError: If a required image is not found in the folder.
    """
    # Read all PNG images
    images = {}
    filenames = {"Isometric": "", "Isometric_rotate": "", "A": "", "B": "", "C": "", "D": ""}
    for file in os.listdir(folder_path):
        if "Isometric" in file:
            filenames[file.split('.')[0]] = file
        elif file.split('_')[-1].split('.')[0] in ["A", "B", "C", "D"]:
            filenames[file.split('_')[-1].split('.')[0]] = file
            
    for key, value in filenames.items():
        img_path = os.path.join(folder_path, value)
        if os.path.exists(img_path):
            images[key] = Image.open(img_path)
        else:
            raise FileNotFoundError(f"Image {value} not found in the folder.")
    
    
    # Get the width and height of all sub-images
    widths = {name: img.width for name, img in images.items()}
    heights = {name: img.height for name, img in images.items()}
    
    # Calculate the total width of the upper and lower rows
    upper_width = widths["Isometric"] + widths["Isometric_rotate"] + spacing
    lower_width = widths["A"] + widths["B"] + widths["C"] + widths["D"] + 3 * spacing
    max_total_width = max(upper_width, lower_width)
    
    upper_height = max(heights["Isometric"], heights["Isometric_rotate"])
    lower_height = max(heights["A"], heights["B"], heights["C"], heights["D"] )
    
    # Calculate the total height of the canvas to ensure all sub-images are fully visible
    total_width = max_total_width + 2 * padding
    total_height = upper_height + lower_height + 3 * padding + 2 * font_size
    
    # Create a white canvas
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    
    # Load font (uses system default font)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    
    # Calculate the starting x position for the upper row
    upper_x = (total_width - upper_width) // 2
    upper_y = padding + font_size
    
    # Place the upper row images
    canvas.paste(images["Isometric"], (upper_x, upper_y))
    canvas.paste(images["Isometric_rotate"], (upper_x + widths["Isometric"] + spacing, upper_y))
    
    # Calculate the starting x position for the lower row
    lower_x = (total_width - lower_width) // 2
    lower_y = upper_y + upper_height + padding + font_size
    
    # Place the lower row images
    canvas.paste(images["A"], (lower_x, lower_y))
    canvas.paste(images["B"], (lower_x + widths["A"] + spacing, lower_y))
    canvas.paste(images["C"], (lower_x + widths["A"] + widths["B"] + 2 * spacing, lower_y))
    canvas.paste(images["D"], (lower_x + widths["A"] + widths["B"] + widths["C"] + 3 * spacing, lower_y))
    
    # Add text above each image and ensure the font is visible
    text_positions = [
        (upper_x + widths["Isometric"] // 2, upper_y - font_size - 10, "View 1"),
        (upper_x + widths["Isometric"] + spacing + widths["Isometric_rotate"] // 2, upper_y - font_size - 10, "View 2"),
        (lower_x + widths["A"] // 2, lower_y - font_size - 10, "A"),
        (lower_x + widths["A"] + spacing + widths["B"] // 2, lower_y - font_size - 10, "B"),
        (lower_x + widths["A"] + widths["B"] + 2 * spacing + widths["C"] // 2, lower_y - font_size - 10, "C"),
        (lower_x + widths["A"] + widths["B"] + widths["C"] + 3 * spacing + widths["D"] // 2, lower_y - font_size - 10, "D"),
    ]
    
    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")
    

def prepare_composite_image_CubeCounting(folder_path, mode):
    """
    Processes cube data from a folder to create a task question.

    Args:
        folder_path (str): The path to the folder containing the cube data.
        mode (int): The mode for creating the composite image.
    """
    image_id = os.path.basename(folder_path)
    pdf_path = os.path.join(folder_path, f"{image_id}_3View.pdf")
    _, x, y, z, num_cube = image_id.split('-')
    
    detect_subfigures(pdf_path, folder_path, padding=10)
    
    view_num = mode + 2
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_CubeCounting(folder_path, output_path, spacing=70, padding=100, font_size=60, mode=mode)
    
    front_grid = detect_grid(os.path.join(folder_path, "Front.png"), int(z), int(x))
    left_grid = detect_grid(os.path.join(folder_path, "Left.png"), int(z), int(y))
    top_grid = detect_grid(os.path.join(folder_path, "Top.png"), int(y), int(x))
    max_num, min_num = get_cube_ans(front_grid, left_grid, top_grid, mode=mode)
    print(f"Number of cubes: {num_cube}; Max: {max_num}; Min: {min_num}")
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    min_max_options = ["at least", "at most", "possibly"]
    explanations = {"ABCD": f"Given {view_num} views, a minimum of {min_num} cubes and a maximum of {max_num} cubes are required."}
    
    num_to_option = {0: "A", 1: "B", 2: "C", 3: "D"}
    min_max_choice = random.choice(min_max_options)
    question = (f"Given {view_num} views, how many cubes are {min_max_choice} needed to satisfy the constraints shown? "
                "Please choose from options A, B, C, or D.")
                
    if min_max_choice == "at least":
        correct_answer_val = min_num
    elif min_max_choice == "at most":
        correct_answer_val = max_num
    else: # possibly
        correct_answer_val = int(num_cube)
        
    candidate_pool = [i for i in range(min_num - 2, max_num + 3) if i != correct_answer_val] + ["None of the above"]
    selected_options = random.sample(candidate_pool, 3)
    choices = [str(correct_answer_val)] + [str(x) for x in selected_options]
    random.shuffle(choices)
    answer = num_to_option[choices.index(str(correct_answer_val))]
    
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
    
    
def prepare_composite_image_CubeAssemply(folder_path, mode=0):
    """
    Processes sliding block puzzle data to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
        mode (int): The mode for the task. Default is 0.
    """
    image_id = os.path.basename(folder_path)
    num_cube = image_id.split("-")[-2]
    num_parts = int(image_id.split("-")[-1])
    png_paths = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    for png_path in png_paths:
        crop_blank_area(os.path.join(folder_path, png_path))
    
    choice_files = [p for p in png_paths if f"compound_{num_parts-1}" in p]
    
    choices = ["A", "B", "C"]
    random.shuffle(choices)
    answer = choices.pop(0)
    
    explanations = {}
    for choice_file in choice_files:
        filename, ext = os.path.splitext(choice_file)
        if "incorrect" in choice_file:
            choice = choices.pop(0)
            os.rename(os.path.join(folder_path, choice_file), os.path.join(folder_path, f"{filename}_{choice}{ext}"))
            explanations[choice] = "Option {choice} is incorrect because it is missing one cube, making the shape of the stack incorrect."
        else:
            os.rename(os.path.join(folder_path, choice_file), os.path.join(folder_path, f"{filename}_{answer}{ext}"))
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_CubeAssembly(folder_path, output_path, spacing=70, padding=100, font_size=60)
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    num_to_word = {1: "one", 2: "two"}
    question = (f"The top-left image shows the original complete cube stack, composed of {num_cube} identical cubes. "
                f"It can be formed by combining the {num_to_word.get(num_parts-1, num_parts-1)} small cube stacks on the right with another small cube stack. "
                "Which of the bottom images is that other cube stack? Please choose from options A, B, C, or D.")
    choices = ["A", "B", "C", "None of the above"]
    
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
    

def prepare_composite_image_CrossSection(folder_path):
    """
    Processes cross-section data of combined solids to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
    """
    image_id = os.path.basename(folder_path)
    png_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
    for pth in png_paths:
        crop_blank_area(pth)
    
    objects = image_id.split('-')[1:]
    object_map = {
        "cube": "square prism", "cylinder": "cylinder", "frustum_circle": "circular frustum",
        "triangular_prism": "regular triangular prism", "frustum_triangle": "regular triangular frustum",
        "frustum_square": "regular square frustum", "cone": "cone",
        "pyramid_triangle": "regular triangular pyramid", "pyramid_square": "regular square pyramid"
    }
    
    obj_caption = "From top to bottom: " + ", ".join([object_map.get(o, o) for o in objects])
    
    incorrect_path = [f for f in os.listdir(folder_path) if "incorrect" in f][0]
    correct_paths = [f for f in os.listdir(folder_path) if "Section" in f and "incorrect" not in f]
    
    random.shuffle(correct_paths)
    
    explanations = {}
    choices = ["A", "B", "C", "D"]
    random.shuffle(choices)
    
    for pth in correct_paths[:3]:
        choice = choices.pop()
        filename, ext = os.path.splitext(pth)
        os.rename(os.path.join(folder_path, pth), os.path.join(folder_path, f"{filename}_{choice}{ext}"))
        
        parts = filename.split('_')
        if parts[1] == "Parallel":
            parallel_face = parts[2]
            explanations[choice] = f"Option {choice} is incorrect. It's a valid cross-section obtained by intersecting the composite solid with a plane parallel to the {parallel_face} plane."
        elif parts[1] == "Rotated":
            faces = {"x": "YZ", "y": "XZ", "z": "XY"}
            angle = parts[2]
            axis = parts[3]
            explanations[choice] = f"Option {choice} is incorrect. It's a valid cross-section obtained by intersecting the solid with a plane perpendicular to the {faces.get(axis, 'unknown')} plane, rotated {angle} degrees around the {axis}-axis."
            
    answer = choices[0]
    filename, ext = os.path.splitext(incorrect_path)
    os.rename(os.path.join(folder_path, incorrect_path), os.path.join(folder_path, f"{filename}_{answer}{ext}"))
    explanations[answer] = "Option {answer} is correct because this cross-section does not match the composite solid shown in the reference images."
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_CrossSection(folder_path, output_path, spacing=70, padding=100, font_size=60)
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    question = (f"The top row shows a composite solid from two different viewpoints. {obj_caption}. "
                "Which of the following images CANNOT be a cross-section of the composite solid? "
                "Please choose from options A, B, C, or D.")
    choices_options = ["A", "B", "C", "D"]
    save_json(category, question_type, level, image_id, question, choices_options, answer, explanations, folder_path)