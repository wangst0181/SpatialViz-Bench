import random, os, json, shutil, time
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import *

def create_incorrect_view_CAD(png_path, mark=[1], mode=0, explanations=None):
    """
    Generates incorrect view options for a CAD drawing.
    Mode 0: Randomly deletes internal lines.
    Mode 1: Randomly flips horizontally or vertically.
    Mode 2: Rotates by 90 degrees.
    Args:
        png_path (str): Path to the correct view's PNG image.
        mark (list, optional): List of markers (e.g., ['B', 'C']) for the generated incorrect options. Defaults to [1].
        mode (int, optional): The mode for generating incorrect views (0, 1, or 2). Defaults to 0.
        explanations (dict, optional): A dictionary to store explanations for why the generated views are incorrect. Defaults to None.
    Returns:
        int: The number of incorrect images generated.
    """
    image = cv2.imread(png_path)
    save_name = png_path.split('.')[0]  
    if mode == 0:
        # Read the image and convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize the image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        first_binary = binary

        external = []
        internal = []
        i = 0
        while True:
            # Find external contours
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if i == 0:
                external.extend(contours)
            else:
                internal.extend(contours)
            if len(contours) == 0:
                break
            
            # Create a blank image to draw the external contours
            external_contours = np.zeros_like(binary)
            
            for j, cnt in enumerate(contours):
                cv2.drawContours(external_contours, [cnt], -1, 255, thickness=10)
            
            # Extract internal lines: original binary image minus the external contours
            internal_lines = cv2.bitwise_and(binary, cv2.bitwise_not(external_contours))
            binary = internal_lines
            i += 1

        if len(internal) == 0:
            return 0
        if len(internal) >= 2: 
            remove_lines = random.sample(internal, 2)
        else:
            remove_lines = random.sample(internal, 1)
        for i, remove_line in enumerate(remove_lines):
            remove_contour = np.zeros_like(binary)
            cv2.drawContours(remove_contour, [remove_line], -1, 255, thickness=2)
            
            reserve_contours= cv2.bitwise_and(first_binary, cv2.bitwise_not(remove_contour))
            
            # Define the structuring element (kernel) for morphological operations
            kernel = np.ones((3, 3), np.uint8)  # Kernel size can be adjusted
            # Opening operation: erosion followed by dilation, to remove thin lines
            processed_reserve_contours = cv2.morphologyEx(reserve_contours, cv2.MORPH_OPEN, kernel)
            result = cv2.bitwise_not(processed_reserve_contours)
            cv2.imwrite(f"{save_name}_{mark[i]}.png", result)
            explanations[mark[i]] = f"Option {mark[i]} is incorrect because the internal outlines are missing."
            if len(mark) == 1:
                return 1
        return len(remove_lines)
    
    elif mode == 1:
        is_symmetric = check_symmetric(image, threshold=0.3)
        if is_symmetric == 'both':
            return 0
        elif is_symmetric == "horizontal":
            flipped_image = cv2.flip(image, 1)
        elif is_symmetric == "vertical":
            flipped_image = cv2.flip(image, 0)
        else:
            flipped_image = cv2.flip(image, random.choice([0,1]))
        cv2.imwrite(f"{save_name}_{mark[0]}.png", flipped_image)
        explanations[mark[0]] = f"Option {mark[0]} is incorrect because the image is a horizontally or vertically mirrored version of an incorrect view."
        return 1

    elif mode == 2:
        Roatation_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f"{save_name}_{mark[0]}.png", Roatation_90)
        explanations[mark[0]] = f"Option {mark[0]} is incorrect because it is the correct view rotated 90 degrees clockwise."
        return 1


def create_composite_image_CAD3View(folder_path, output_path, spacing=70, padding=100, font_size=60):
    """
    Creates a composite image for a CAD problem from individual view images.
    Args:
        folder_path (str): The folder containing the source PNG images (e.g., 'Isometric.png', 'ref.png', 'A.png').
        output_path (str): The path to save the final composite image.
        spacing (int, optional): The space between images. Defaults to 70.
        padding (int, optional): The padding around the entire canvas. Defaults to 100.
        font_size (int, optional): The font size for the labels. Defaults to 60.
    Raises:
        FileNotFoundError: If a required image is not found in the folder.
    """
    # Read all PNG images
    images = {}
    filenames = {"Isometric": "", "ref": "", "A": "", "B": "", "C": ""}
    for file in os.listdir(folder_path):
        if file.split('_')[-1].split('.')[0] in filenames:
            filenames[file.split('_')[-1].split('.')[0]] = file

    for key, value in filenames.items():
        img_path = os.path.join(folder_path, value)
        if os.path.exists(img_path):
            images[key] = Image.open(img_path)
        else:
            raise FileNotFoundError(f"Image {value} not found in the folder.")
    
    images["ref"] = resize_image_keep_aspect_pil(images["ref"], images["Isometric"].height)
    
    # Get the width and height of all subfigures
    widths = {name: img.width for name, img in images.items()}
    heights = {name: img.height for name, img in images.items()}
    
    # Calculate the total width of the upper and lower rows
    upper_width = widths["Isometric"] + widths["ref"] + spacing
    lower_width = widths["A"] + widths["B"] + widths["C"] + 2 * spacing
    max_total_width = max(upper_width, lower_width)
    
    upper_height = max(heights["Isometric"], heights["ref"])
    lower_height = max(heights["A"], heights["B"], heights["C"])
    
    # Calculate the total canvas height to ensure all subfigures are fully visible
    total_width = max_total_width + 2 * padding
    total_height = upper_height + lower_height + 3 * padding + 2 * font_size
    
    # Create a white canvas
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    
    # Load font (use a system default if 'DejaVuSans-Bold.ttf' is not available)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate the starting x position for the upper row
    upper_x = (total_width - upper_width) // 2
    upper_y = padding + font_size
    
    # Place the upper row images
    canvas.paste(images["Isometric"], (upper_x, upper_y))
    canvas.paste(images["ref"], (upper_x + widths["Isometric"] + spacing, upper_y))
    
    # Calculate the starting x position for the lower row
    lower_x = (total_width - lower_width) // 2
    lower_y = upper_y + upper_height + padding + font_size
    
    # Place the lower row images
    canvas.paste(images["A"], (lower_x, lower_y))
    canvas.paste(images["B"], (lower_x + widths["A"] + spacing, lower_y))
    canvas.paste(images["C"], (lower_x + widths["A"] + widths["B"] + 2 * spacing, lower_y))
    
    # Add text above each image and ensure the font is visible
    text_positions = [
        (upper_x + widths["Isometric"] // 2, upper_y - font_size - 10, "Isometric"),
        (upper_x + widths["Isometric"] + spacing + widths["ref"] // 2, upper_y - font_size - 10, filenames["ref"].split('_')[0]),
        (lower_x + widths["A"] // 2, lower_y - font_size - 10, "A"),
        (lower_x + widths["A"] + spacing + widths["B"] // 2, lower_y - font_size - 10, "B"),
        (lower_x + widths["A"] + widths["B"] + 2 * spacing + widths["C"] // 2, lower_y - font_size - 10, "C")
    ]
    
    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")
    

def create_composite_image_Cubes3View(folder_path, output_path, spacing=70, padding=100, font_size=60):
    """
    Creates a composite reference image for cube stack 3-view problems.

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
    filenames = {"Isometric": "", "Front": "", "Top": "", "A": "", "B": "", "C": "", "D": ""}
    for file in os.listdir(folder_path):
        filenames[file.split('_')[-1].split('.')[0]] = file
    print(filenames)

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
    upper_width = widths["Isometric"] + widths["Front"] + widths["Top"] + 2 * spacing
    lower_width = widths["A"] + widths["B"] + widths["C"] + widths["D"] + 3 * spacing
    max_total_width = max(upper_width, lower_width)
    
    upper_height = max(heights["Isometric"], heights["Front"], heights["Top"])
    lower_height = max(heights["A"], heights["B"], heights["C"], heights["D"])
    
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
    canvas.paste(images["Front"], (upper_x + widths["Isometric"] + spacing, upper_y))
    canvas.paste(images["Top"], (upper_x + widths["Isometric"] + widths["Front"] + 2 * spacing, upper_y))
    
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
        (upper_x + widths["Isometric"] // 2, upper_y - font_size - 10, "Isometric"),
        (upper_x + widths["Isometric"] + spacing + widths["Front"] // 2, upper_y - font_size - 10, "Front"),
        (upper_x + widths["Isometric"] + widths["Front"] + 2 * spacing + widths["Top"] // 2, upper_y - font_size - 10, "Top"),
        (lower_x + widths["A"] // 2, lower_y - font_size - 10, "A"),
        (lower_x + widths["A"] + spacing + widths["B"] // 2, lower_y - font_size - 10, "B"),
        (lower_x + widths["A"] + widths["B"] + 2 * spacing + widths["C"] // 2, lower_y - font_size - 10, "C"),
        (lower_x + widths["A"] + widths["B"] + widths["C"] + 3 * spacing + widths["D"] // 2, lower_y - font_size - 10, "D")
    ]
    
    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")
    

def create_composite_image_Roatation(folder_path, output_path, spacing=70, padding=100, font_size=60):
    """
    Creates a composite reference image for 2D rotation problems.

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
    filenames = {"original": "","A": "", "B": "", "C": ""}
    for file in os.listdir(folder_path):
        print(file)
        if file.split('_')[-1].split('.')[0] in filenames:
            filenames[file.split('_')[-1].split('.')[0]] = file            
    print(filenames)
    
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
    upper_width = widths["original"] + widths["A"] + widths["B"] + widths["C"]+ 3 * spacing
    upper_height = max(heights["original"], heights["A"], heights["B"], heights["C"])
    
    # Calculate the total height of the canvas to ensure all sub-images are fully visible
    total_width = upper_width + 2 * padding
    total_height = upper_height + 2 * padding +  font_size
    
    # Create a white canvas
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    
    # Load font (uses system default font)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    
    # Calculate the starting x position for the upper row
    upper_x = (total_width - upper_width) // 2
    upper_y = padding + font_size
    
    # Place the upper row images
    canvas.paste(images["original"], (upper_x, upper_y))
    canvas.paste(images["A"], (upper_x + widths["original"] + spacing, upper_y))
    canvas.paste(images["B"], (upper_x + widths["original"] + widths["A"] + 2*spacing, upper_y))
    canvas.paste(images["C"], (upper_x + widths["original"] + widths["A"] + widths["B"] + 3*spacing, upper_y))
    
    # Add text above each image and ensure the font is visible
    text_positions = [
        (upper_x + widths["original"] // 2, upper_y - font_size - 10, "Original"),
        (upper_x + widths["original"] + spacing + widths["A"] // 2, upper_y - font_size - 10, "A"),
        (upper_x + widths["original"] + widths["A"] + 2 * spacing + widths["B"] // 2, upper_y - font_size - 10, "B"),
        (upper_x + widths["original"] + widths["A"] + widths["B"] + 3 * spacing + widths["C"] // 2, upper_y - font_size - 10, "C")
    ]
    
    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")


def prepare_composite_image_CAD(folder_path):
    """
    Processes CAD data from a folder to create a task question.

    Args:
        folder_path (str): The path to the folder containing CAD data.
    """
    image_id = os.path.basename(folder_path)
    pdf_path = os.path.join(folder_path, f"{image_id}_3View.pdf")
    png_path = os.path.join(folder_path, f"{image_id}_Isometric.png")

    if detect_subfigures(pdf_path, folder_path, padding=10) is None:
        return None
    crop_blank_area(png_path)
    
    views = ["Front", "Left", "Top"]
    ref_view = find_max_area(folder_path)
    views.remove(ref_view)
    choice_view = random.choice(views)
    
    choices = ["A", "B", "C"]
    random.shuffle(choices)
    answer = choices[0]
    
    explanations = {}

    num_incorrect = create_incorrect_view_CAD(
        os.path.join(folder_path, f"{choice_view}.png"),
        mark=choices[1:],
        mode=0,
        explanations=explanations
    )

    remaining_views = [v for v in views if v != choice_view]
    
    if num_incorrect < 2:
        num_incorrect += create_incorrect_view_CAD(
            os.path.join(folder_path, f"{remaining_views[0]}.png"),
            mark=choices[num_incorrect+1:],
            mode=0,
            explanations=explanations
        )
    
    i = 1
    while num_incorrect < 2:
        num_incorrect += create_incorrect_view_CAD(
            os.path.join(folder_path, f"{remaining_views[0]}.png"),
            mark=[choices[2]],
            mode=i,
            explanations=explanations
        )
        i += 1
        if i > 2:
            return None
            
    if num_incorrect == 0:
        return None
    
    print(explanations)

    os.rename(
        os.path.join(folder_path, f"{choice_view}.png"),
        os.path.join(folder_path, f"{choice_view}_{answer}.png")
    )
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_CAD3View(folder_path, output_path, spacing=70, padding=100, font_size=60)
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    
    view_name_map = {"Front": "Front View", "Left": "Left View", "Top": "Top View"}
    question = (f"The upper-left image is an isometric view of a 3D model, and the right image is the {view_name_map[ref_view]}. "
                f"Which of the lower images is the model's {view_name_map[choice_view]}? "
                "Please choose from options A, B, C, or D.")
    
    choices = ["A", "B", "C", "None of the above"]
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
    return 'success'


def prepare_composite_image_Cubes3View(folder_path):
    """
    Processes cube counting data from a folder to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
    """
    image_id = os.path.basename(folder_path)
    png_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    for png_path in png_paths:
        crop_blank_area(png_path)

    incorrect_paths = [p for p in png_paths if "incorrect" in p or "Right" in p]
    correct_path = os.path.join(folder_path, "cubes_Left.png")
    
    choices = ["A", "B", "C", "D"]
    random.shuffle(choices)
    answer = choices[0]
    os.rename(correct_path, os.path.join(folder_path, f"cubes_Left_{answer}.png"))
    
    explanations = {}
    for i, incorrect_path in enumerate(incorrect_paths):
        new_filename = f"{os.path.splitext(incorrect_path)[0]}_{choices[i+1]}.png"
        os.rename(incorrect_path, new_filename)
        
        if "incorrect_Right" in incorrect_path:
            explanations[choices[i+1]] = (f"Option {choices[i+1]} is incorrect because it shows the Right View shape, not the Left View, "
                                           "and the position of the red cube is also incorrect.")
        elif "incorrect_Left" in incorrect_path:
            explanations[choices[i+1]] = f"Option {choices[i+1]} is incorrect because the position of the red cube in the view is incorrect."
        elif "cubes_Right" in incorrect_path:
            explanations[choices[i+1]] = f"Option {choices[i+1]} is incorrect because it is the Right View of the cube stack, not the Left View."
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_Cubes3View(folder_path, output_path, spacing=70, padding=100, font_size=60)
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    question = ("The image shows a stack of identical small cubes. Most cubes are gray, and a few are red. "
                "The top row, from left to right, shows the isometric, front, and top views of the cube stack. "
                "Which of the bottom images is the Left View of the cube stack? Please choose from options A, B, C, or D.")
    choices = ["A", "B", "C", "D"]
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
   

def prepare_composite_image_2DRotation(folder_path):
    """
    Processes 2D rotation data to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
    """
    image_id = os.path.basename(folder_path)
    png_paths = os.listdir(folder_path)
    for png_path in png_paths:
        crop_blank_area(os.path.join(folder_path, png_path))
    original_png_path = [p for p in png_paths if "transform" not in p][0]
    
    rotation, h_rotation, v_rotation = rotate_and_flip_figure(os.path.join(folder_path, original_png_path))
    
    correct_view_file = [f for f in os.listdir(folder_path) if "rotated" in f][0]
    incorrect_view_files = [f for f in os.listdir(folder_path) if "horizontal" in f or "vertical" in f or "transform" in f]
    incorrect_view_files = random.sample(incorrect_view_files, 2)
    
    explanations = {}
    choices = ["A", "B", "C"]
    random.shuffle(choices)
    answer = choices[0]
    filename, ext = os.path.splitext(correct_view_file)
    os.rename(os.path.join(folder_path, correct_view_file), os.path.join(folder_path, f"{filename}_{answer}{ext}"))
    explanations[answer] = f"Option {answer} is correct because it is obtained by rotating the original image by {rotation} degrees."
    
    for i, incorrect_file in enumerate(incorrect_view_files):
        filename, ext = os.path.splitext(incorrect_file)
        os.rename(os.path.join(folder_path, incorrect_file), os.path.join(folder_path, f"{filename}_{choices[i+1]}{ext}"))
        if "horizontal" in incorrect_file:
            explanations[choices[i+1]] = f"Option {choices[i+1]} is incorrect because it is obtained by rotating the original image by {h_rotation} degrees and then flipping it horizontally."
        elif "vertical" in incorrect_file:
            explanations[choices[i+1]] = f"Option {choices[i+1]} is incorrect because it is obtained by rotating the original image by {v_rotation} degrees and then flipping it vertically."
        elif "transform" in incorrect_file:
            explanations[choices[i+1]] = "Option {choices[i+1]} is incorrect because the non-centrosymmetric pattern within it has been rotated."
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_Roatation(folder_path, output_path, spacing=70, padding=100, font_size=60)
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    question = ("The left image shows a colored grid paper, with one corner marked by a red square. "
                "Which of the grid papers in the options can be obtained by only rotating the grid paper on the left? "
                "Please choose from options A, B, C, or D.")
    choices = ["A", "B", "C", "None of the above"]
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)


def prepare_composite_image_3DRotation(folder_path, mode=0):
    """
    Processes 3D rotation data to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
        mode (int): The mode for the task. 0 for 'cannot be obtained by rotation', 1 for 'can be obtained by rotation'.
    """
    image_id = os.path.basename(folder_path)
    png_paths = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    for png_path in png_paths:
        crop_blank_area(os.path.join(folder_path, png_path), padding=10)
    
    explanations = {}
    choices = ["A", "B", "C"]
    random.shuffle(choices)
    answer = choices[0]

    output_path = os.path.join(folder_path, f"{image_id}.png")
    
    if mode == 1:
        png_path = os.path.join(folder_path, "original.png")
        image = cv2.imread(png_path)
        if image is not None:
            cv2.imwrite(os.path.join(folder_path, "horizontal.png"), cv2.flip(image, 1)) # 1 for horizontal flip
            cv2.imwrite(os.path.join(folder_path, "vertical.png"), cv2.flip(image, 0))   # 0 for vertical flip
        
        all_pngs = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        correct_files = [f for f in all_pngs if "original" not in f and "remove" not in f and "vertical" not in f and "horizontal" not in f]
        incorrect_files = [f for f in all_pngs if "vertical" in f or "horizontal" in f]
    
        correct_file = random.choice(correct_files)
        filename, ext = os.path.splitext(correct_file)
        os.rename(os.path.join(folder_path, correct_file), os.path.join(folder_path, f"{filename}_{answer}{ext}"))
        axis, angle = filename.split('-')
        explanations[answer] = f"Option {answer} is correct. It's the original stack rotated {angle} degrees around the {axis}-axis."
        
        flip_map = {'horizontal': 'horizontally', 'vertical': 'vertically'}
        incorrect_file = random.choice(incorrect_files)
        filename, ext = os.path.splitext(incorrect_file)
        os.rename(os.path.join(folder_path, incorrect_file), os.path.join(folder_path, f"{filename}_{choices[1]}{ext}"))
        explanations[choices[1]] = f"Option {choices[1]} is incorrect. It's a mirrored version of the original stack, flipped {flip_map.get(filename)}."
        
        remove_file = [f for f in all_pngs if "remove" in f][0]
        filename, ext = os.path.splitext(remove_file)
        os.rename(os.path.join(folder_path, remove_file), os.path.join(folder_path, f"{filename}_{choices[2]}{ext}"))
        explanations[choices[2]] = "Option {choices[2]} is incorrect. It's the original stack with one cube removed."
        
        question = ("The left image is the original cube stack, made of identical cubes. "
                    "Which of the cube stacks in the options can be obtained by only rotating the original cube stack? "
                    "Please choose from options A, B, C, or D.")
    else: # mode == 0
        correct_file = [f for f in png_paths if "remove" in f][0]
        incorrect_files = [f for f in png_paths if "remove" not in f and "original" not in f]
        
        filename, ext = os.path.splitext(correct_file)
        os.rename(os.path.join(folder_path, correct_file), os.path.join(folder_path, f"{filename}_{answer}{ext}"))
        explanations[answer] = "Option {answer} is correct. This stack cannot be obtained by rotating the original; it has one cube removed."
        
        selected_incorrect = random.sample(incorrect_files, 2)
        for i, incorrect_file in enumerate(selected_incorrect):
            filename, ext = os.path.splitext(incorrect_file)
            os.rename(os.path.join(folder_path, incorrect_file), os.path.join(folder_path, f"{filename}_{choices[i+1]}{ext}"))
            axis, angle = filename.split('-')
            explanations[choices[i+1]] = f"Option {choices[i+1]} is incorrect. This stack can be obtained by rotating the original {angle} degrees around the {axis}-axis."

        question = ("The left image is the original cube stack, made of identical cubes. "
                    "Which of the cube stacks in the options CANNOT be obtained by only rotating the original cube stack? "
                    "Please choose from options A, B, C, or D.")

    create_composite_image_Roatation(folder_path, output_path, spacing=70, padding=100, font_size=60)
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    choices_options = ["A", "B", "C", "None of the above"]
    save_json(category, question_type, level, image_id, question, choices_options, answer, explanations, folder_path)