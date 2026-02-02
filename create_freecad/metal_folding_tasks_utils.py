import random, os, json, shutil, time
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import *


def create_incorrect_view_reconstruct(folder_path, png_selected, incorrect_choices, explanation):
    """
    Generates incorrect reconstruction views by flipping images.
    Args:
        folder_path (str): The folder containing the PNG images.
        png_selected (list): A list of selected PNG file names to be modified.
        incorrect_choices (list): The choice labels (e.g., 'B', 'C') for the incorrect options.
        explanation (dict): A dictionary to store explanations for the incorrect views.
    """
    for i, png_name in enumerate(png_selected):
        png_path = f"{folder_path}/{png_name}"
        crop_blank_area(png_path)
        image = cv2.imread(png_path)
        flipped_id = random.choice([0,1]) # 0 for vertical flip, 1 for horizontal
        flipped_directions = ["vertically", "horizontally"]
        flipped_image = cv2.flip(image, flipped_id)
        cv2.imwrite(f"{png_path.split('.')[0]}_incorrect_{incorrect_choices[i]}.png", flipped_image)
        views = {
            "FrontTopRight": "Front-Top-Right",
            "FrontTopLeft": "Front-Top-Left",
            "FrontBottomRight": "Front-Bottom-Right",
            "FrontBottomLeft": "Front-Bottom-Left",
            "BackTopRight": "Back-Top-Right",
            "BackTopLeft": "Back-Top-Left",
            "BackBottomRight": "Back-Bottom-Right",
            "BackBottomLeft": "Back-Bottom-Left",
        }
        explanation[incorrect_choices[i]] = f"Assuming the base is the square in the 2nd row, 1st column of the net, and the right face is the one to the right of the base. Option {incorrect_choices[i]} is incorrect because it is the {views[png_name.split('.')[0]]} view mirrored {flipped_directions[flipped_id]}."
        
        
def create_composite_image_CubeReconstruction(folder_path, output_path, spacing=70, padding=100, font_size=60):
    """
    Creates a composite reference image for cube net reconstruction problems.

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
    filenames = {"unfold": "","A": "", "B": "", "C": "", "D": ""}
    for file in os.listdir(folder_path):
        if file.split('_')[-1].split('.')[0] in filenames:
            filenames[file.split('_')[-1].split('.')[0]] = file            
    crop_blank_area(os.path.join(folder_path, filenames["unfold"]))
    
    for key, value in filenames.items():
        if not value:
            continue
        img_path = os.path.join(folder_path, value)
        if os.path.exists(img_path):
            images[key] = Image.open(img_path)
        else:
            raise FileNotFoundError(f"Image {value} not found in the folder.")
    
    images["unfold"] = resize_image_keep_aspect_pil(images["unfold"], images["A"].height)
    
    # Get the width and height of all sub-images
    widths = {name: img.width for name, img in images.items()}
    heights = {name: img.height for name, img in images.items()}
    
    # Calculate the total width of the upper and lower rows
    if not filenames["D"]:
        upper_width = widths["unfold"] + widths["A"] + widths["B"] + widths["C"]+ 3 * spacing
        upper_height = max(heights["unfold"], heights["A"], heights["B"], heights["C"])
    else:
        upper_width = widths["unfold"] + widths["A"] + widths["B"] + widths["C"] + widths["D"] + 4 * spacing
        upper_height = max(heights["unfold"], heights["A"], heights["B"], heights["C"], heights["D"])
    
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
    canvas.paste(images["unfold"], (upper_x, upper_y))
    canvas.paste(images["A"], (upper_x + widths["unfold"] + spacing, upper_y))
    canvas.paste(images["B"], (upper_x + widths["unfold"] + widths["A"] + 2*spacing, upper_y))
    canvas.paste(images["C"], (upper_x + widths["unfold"] + widths["A"] + widths["B"] + 3*spacing, upper_y))
    if filenames["D"]:
        canvas.paste(images["D"], (upper_x + widths["unfold"] + widths["A"] + widths["B"] + widths["C"] + 4*spacing, upper_y))
    
    # Add text above each image and ensure the font is visible
    text_positions = [
        (upper_x + widths["unfold"] // 2, upper_y - font_size - 10, "Cube Net"),
        (upper_x + widths["unfold"] + spacing + widths["A"] // 2, upper_y - font_size - 10, "A"),
        (upper_x + widths["unfold"] + widths["A"] + 2 * spacing + widths["B"] // 2, upper_y - font_size - 10, "B"),
        (upper_x + widths["unfold"] + widths["A"] + widths["B"] + 3 * spacing + widths["C"] // 2, upper_y - font_size - 10, "C")
    ]
    if filenames["D"]:
        text_positions.append(
            (upper_x + widths["unfold"] + widths["A"] + widths["B"] + widths["C"] + 4 * spacing + widths["D"] // 2, upper_y - font_size - 10, "D")
        )
    
    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")
    
    
def create_composite_image_CubeUnfolding(folder_path, output_path, spacing=70, padding=100, font_size=60):
    """
    Creates a composite reference image for cube unfolding problems.

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
    filenames = {"cube": "","A": "", "B": "", "C": "", "D": ""}
    for file in os.listdir(folder_path):
        if file.split('_')[-1].split('.')[0] in filenames:
            filenames[file.split('_')[-1].split('.')[0]] = file            
    
    for key, value in filenames.items():
        img_path = os.path.join(folder_path, value)
        if os.path.exists(img_path):
            images[key] = Image.open(img_path)
        else:
            raise FileNotFoundError(f"Image {value} not found in the folder.")
        
    images["cube"] = resize_image_keep_aspect_pil(images["cube"], images["A"].height)
    
    # Get the width and height of all sub-images
    widths = {name: img.width for name, img in images.items()}
    heights = {name: img.height for name, img in images.items()}
    
    # Calculate the total width of the upper and lower rows
    upper_width = widths["cube"] + widths["A"] + widths["B"] + widths["C"]+ widths["D"] + 4 * spacing
    upper_height = max(heights["cube"], heights["A"], heights["B"], heights["C"], widths["D"])
    
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
    canvas.paste(images["cube"], (upper_x, upper_y))
    canvas.paste(images["A"], (upper_x + widths["cube"] + spacing, upper_y))
    canvas.paste(images["B"], (upper_x + widths["cube"] + widths["A"] + 2*spacing, upper_y))
    canvas.paste(images["C"], (upper_x + widths["cube"] + widths["A"] + widths["B"] + 3*spacing, upper_y))
    canvas.paste(images["D"], (upper_x + widths["cube"] + widths["A"] + widths["B"] + widths["C"] + 4*spacing, upper_y))
    
    # Add text above each image and ensure the font is visible
    text_positions = [
        (upper_x + widths["cube"] // 2, upper_y - font_size - 10, "Cube"),
        (upper_x + widths["cube"] + spacing + widths["A"] // 2, upper_y - font_size - 10, "A"),
        (upper_x + widths["cube"] + widths["A"] + 2 * spacing + widths["B"] // 2, upper_y - font_size - 10, "B"),
        (upper_x + widths["cube"] + widths["A"] + widths["B"] + 3 * spacing + widths["C"] // 2, upper_y - font_size - 10, "C"),
        (upper_x + widths["cube"] + widths["A"] + widths["B"] + widths["C"] + 4 * spacing + widths["D"] // 2, upper_y - font_size - 10, "D")
    ]
    
    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")
    

def create_composite_image_PaperFolding(folder_path, output_path, spacing=70, padding=100, font_size=60, ops=3):
    """
    Creates a composite reference image for paper folding problems.

    Args:
        folder_path (str): The path to the folder containing the source images.
        output_path (str): The path to save the composite image.
        spacing (int, optional): The spacing between images. Defaults to 70.
        padding (int, optional): The padding around the canvas. Defaults to 100.
        font_size (int, optional): The font size for the labels. Defaults to 60.
        ops (int, optional): The number of folding operations (2 or 3). Defaults to 3.

    Raises:
        FileNotFoundError: If a required image is not found in the folder.
    """
    # Read all PNG images
    print(ops)
    images = {}
    filenames = {"1": "", "2": "", "3": "", "4": "", "A": "", "B": "", "C": "", "D": ""}
    for file in os.listdir(folder_path):
        if file.split('_')[-1].split('.')[0] in ["horizontal", "vertical", "diagonal", "punch"]:
            filenames[file.split('_')[0]] = file
        elif "unfold" in file:
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
    if ops == 3:
        upper_width = widths["1"] + widths["2"] + widths["3"] + widths["4"] + 3 * spacing
        upper_height = max(heights["1"], heights["2"], heights["3"], heights["4"])
    elif ops == 2:
        upper_width = widths["1"] + widths["2"] + widths["3"] + 2 * spacing
        upper_height = max(heights["1"], heights["2"], heights["3"])
    lower_width = widths["A"] + widths["B"] + widths["C"] + widths["D"] + 3 * spacing
    max_total_width = max(upper_width, lower_width)
    
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
    canvas.paste(images["1"], (upper_x, upper_y))
    canvas.paste(images["2"], (upper_x + widths["1"] + spacing, upper_y))
    canvas.paste(images["3"], (upper_x + widths["1"] +  widths["2"] + 2 * spacing, upper_y))
    if ops == 3:
        canvas.paste(images["4"], (upper_x + widths["1"] +  widths["2"] + widths["3"] + 3 * spacing, upper_y))
    
    # Calculate the starting x position for the lower row
    lower_x = (total_width - lower_width) // 2
    lower_y = upper_y + upper_height + padding + font_size
    
    # Place the lower row images
    canvas.paste(images["A"], (lower_x, lower_y))
    canvas.paste(images["B"], (lower_x + widths["A"] + spacing, lower_y))
    canvas.paste(images["C"], (lower_x + widths["A"] + widths["B"] + 2 * spacing, lower_y))
    canvas.paste(images["D"], (lower_x + widths["A"] + widths["B"] + widths["C"] + 3 * spacing, lower_y))
    
    # Add text above each image and ensure the font is visible
    if ops == 3:
        text_positions = [
            (upper_x + widths["1"] // 2, upper_y - font_size - 10, "Operation 1"),
            (upper_x + widths["1"] + spacing + widths["2"] // 2, upper_y - font_size - 10, "Operation 2"),
            (upper_x + widths["1"] + widths["2"] + 2 * spacing + widths["3"] // 2, upper_y - font_size - 10, "Operation 3"),
            (upper_x + widths["1"] + widths["2"] + widths["3"] + 3 * spacing + widths["4"] // 2, upper_y - font_size - 10, "Punch Holes"),
            (lower_x + widths["A"] // 2, lower_y - font_size - 10, "A"),
            (lower_x + widths["A"] + spacing + widths["B"] // 2, lower_y - font_size - 10, "B"),
            (lower_x + widths["A"] + widths["B"] + 2 * spacing + widths["C"] // 2, lower_y - font_size - 10, "C"),
            (lower_x + widths["A"] + widths["B"] + widths["C"] + 3 * spacing + widths["D"] // 2, lower_y - font_size - 10, "D"),
        ]
    elif ops == 2:
        text_positions = [
            (upper_x + widths["1"] // 2, upper_y - font_size - 10, "Operation 1"),
            (upper_x + widths["1"] + spacing + widths["2"] // 2, upper_y - font_size - 10, "Operation 2"),
            (upper_x + widths["1"] + widths["2"] + 2 * spacing + widths["3"] // 2, upper_y - font_size - 10, "Punch Holes"),
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
    
    
def prepare_composite_image_ReconstructCube(folder_path):
    """
    Processes cube net data from a folder to create a task question.

    Args:
        folder_path (str): The path to the folder containing the cube net data.
    """
    image_id = os.path.basename(folder_path)
    png_files = [f for f in os.listdir(folder_path) if "Back" in f or "Front" in f]
    selected_view_files = random.sample(png_files, 3)
    correct_view_file = selected_view_files[0]
    incorrect_view_files = selected_view_files[1:]
    crop_blank_area(os.path.join(folder_path, correct_view_file))
    
    with open(os.path.join(folder_path, "info.json"), 'r', encoding="utf-8") as f:
        info = json.load(f)
    
    choices = ["A", "B", "C"]
    random.shuffle(choices)
    answer = choices[0]
    incorrect_choices = choices[1:]
    
    views = {
        "FrontTopRight": "Front-Top-Right", "FrontTopLeft": "Front-Top-Left",
        "FrontBottomRight": "Front-Bottom-Right", "FrontBottomLeft": "Front-Bottom-Left",
        "BackTopRight": "Back-Top-Right", "BackTopLeft": "Back-Top-Left",
        "BackBottomRight": "Back-Bottom-Right", "BackBottomLeft": "Back-Bottom-Left",
    }
    color_map = {"red": "Red", "green": "Green", "blue": "Blue", "pink": "Pink", "yellow": "Yellow", "cyan": "Cyan"}
    explanations = {}
    
    create_incorrect_view_reconstruct(folder_path, incorrect_view_files, incorrect_choices, explanations)
    
    correct_filename, correct_ext = os.path.splitext(correct_view_file)
    os.rename(os.path.join(folder_path, correct_view_file), os.path.join(folder_path, f"{correct_filename}_{answer}{correct_ext}"))
    
    explanation_text = (f"After folding the net, the front face is {color_map[info['Front']['color'][0]]}, "
                        f"the back is {color_map[info['Back']['color'][0]]}, the left is {color_map[info['Left']['color'][0]]}, "
                        f"the right is {color_map[info['Right']['color'][0]]}, the top is {color_map[info['Top']['color'][0]]}, "
                        f"and the bottom is {color_map[info['Bottom']['color'][0]]}.")
    explanations["ABCD"] = explanation_text
    
    correct_view_name = os.path.splitext(correct_view_file)[0]
    explanations[answer] = f"Option {answer} is correct because it shows the {views.get(correct_view_name, 'unknown')} view."
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    
    if random.choice([True, False]):
        output_path = os.path.join(folder_path, f"{image_id}.png")
        create_composite_image_CubeReconstruction(folder_path, output_path, spacing=70, padding=100, font_size=60)
        
        question = ("The left image is the net of a small cube, with its six faces painted different colors. "
                    "The net is folded upwards to form the cube. "
                    "Which of the following adjacent color combinations is possible when viewing the cube from an isometric perspective? "
                    "Please choose from options A, B, C, or D.")
        choices = ["A", "B", "C", "None of the above"]
        save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
    else:
        os.rename(os.path.join(folder_path, f"{image_id}_unfold.png"), os.path.join(folder_path, f"{image_id}.png"))
        option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        opposite_views = {"Bottom": "Top", "Top": "Bottom", "Left": "Right", "Right": "Left", "Front": "Back", "Back": "Front"}
        
        view = random.choice(list(opposite_views.keys()))
        opposite_view = opposite_views[view]
        
        view_color_name = info[view]["color"][0]
        opposite_view_color_name = info[opposite_view]["color"][0]
        
        view_color_display = color_map.pop(view_color_name)
        opposite_view_color_display = color_map.pop(opposite_view_color_name)
        
        choices = [opposite_view_color_display] + random.sample(list(color_map.values()), 2) + ["None of the above"]
        random.shuffle(choices)
        
        question = (f"The image shows the net of a small cube with its six faces painted different colors. "
                    f"The net is folded upwards to form the cube. What color is on the face opposite the {view_color_display} face? "
                    "Please choose from options A, B, C, or D.")
        
        answer_index = choices.index(opposite_view_color_display)
        answer = option_map[answer_index]
        
        explanations["ABCD"] = explanation_text
        save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)


def prepare_composite_image_ReconstructCubeComplex(folder_path):
    """
    Processes complex cube net data from a folder to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
    """
    image_id = os.path.basename(folder_path)
    correct_view_files = [f for f in os.listdir(folder_path) if "incorrect" not in f and "unfold" not in f]
    incorrect_view_files = [f for f in os.listdir(folder_path) if "incorrect" in f]
    
    choices = ["A", "B", "C", "D"]
    random.shuffle(choices)
    answer = choices[0]
    
    views = {
        "FrontTopRight": "Front-Top-Right", "FrontTopLeft": "Front-Top-Left",
        "FrontBottomRight": "Front-Bottom-Right", "FrontBottomLeft": "Front-Bottom-Left",
        "BackTopRight": "Back-Top-Right", "BackTopLeft": "Back-Top-Left",
        "BackBottomRight": "Back-Bottom-Right", "BackBottomLeft": "Back-Bottom-Left",
    }
    
    explanations = {}    
    correct_view_file = random.choice(correct_view_files)
    correct_view_files.remove(correct_view_file)
    crop_blank_area(os.path.join(folder_path, correct_view_file))
    
    correct_filename, correct_ext = os.path.splitext(correct_view_file)
    os.rename(os.path.join(folder_path, correct_view_file), os.path.join(folder_path, f"{correct_filename}_{answer}{correct_ext}"))
    
    view_name = views.get(correct_filename, "unknown")
    explanations[answer] = (f"Assuming the base is the square in the 2nd row, 1st column of the net, and the right face is the square to its right. "
                            f"Option {answer} is correct because it shows the {view_name} view.")
    
    num_rotated_incorrect = random.randint(1, 2)
    incorrect_rotated_files = random.sample(incorrect_view_files, num_rotated_incorrect)
    for i, file in enumerate(incorrect_rotated_files):
        crop_blank_area(os.path.join(folder_path, file))
        filename, ext = os.path.splitext(file)
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, f"{filename}_{choices[i+1]}{ext}"))
        explanations[choices[i+1]] = (f"Assuming the base is the square in the 2nd row, 1st column of the net, and the right face is the square to its right. "
                                       f"Option {choices[i+1]} is incorrect because the face with the non-centrosymmetric pattern has been rotated.")
        
    create_incorrect_view_reconstruct(folder_path, random.sample(correct_view_files, 3-num_rotated_incorrect), choices[num_rotated_incorrect+1:], explanations)
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_CubeReconstruction(folder_path, output_path, spacing=70, padding=100, font_size=60)
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    question = ("The left image is the net of a small cube, with its six faces having different patterns. "
                "The net is folded upwards to form the cube. "
                "Which of the following adjacent pattern combinations is possible when viewing the cube from an isometric perspective? "
                "Please choose from options A, B, C, or D.")
    choices = ["A", "B", "C", "D"]
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
   
    
def prepare_composite_image_CubeUnfolding(folder_path):
    """
    Processes unfolded cube data to create a task question.

    Args:
        folder_path (str): The path to the folder with the data.
    """
    image_id = os.path.basename(folder_path)
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    choices = ["A", "B", "C", "D"]
    random.shuffle(choices)
    answer = choices.pop(0)
    
    explanations = {}
    cube_file = [f for f in os.listdir(folder_path) if 'cube' in f][0]
    colors = os.path.splitext(cube_file)[0].split('_')[:3]
    color_map = {"red": "Red", "green": "Green", "blue": "Blue", "pink": "Pink", "yellow": "Yellow", "cyan": "Cyan"}
    display_colors = [color_map.get(c, c) for c in colors]
    
    remaining_choices = "".join(choices)
    explanations[remaining_choices] = (f"Options {remaining_choices} are incorrect because they could be valid nets for the given cube. "
                                       f"The positions of the {display_colors[0]}, {display_colors[1]}, and {display_colors[2]} faces "
                                       "satisfy the relative positions shown on the cube.")
    
    for png_file in png_files:
        crop_blank_area(os.path.join(folder_path, png_file))
        filename, ext = os.path.splitext(png_file)
        if '_correct_' in png_file:
            choice = choices.pop(0)
            os.rename(os.path.join(folder_path, png_file), os.path.join(folder_path, f"{filename}_{choice}{ext}"))
        elif '_incorrect_' in png_file:
            os.rename(os.path.join(folder_path, png_file), os.path.join(folder_path, f"{filename}_{answer}{ext}"))
            color0, color1 = filename.split('_')[-1].split('-')
            explanations[answer] = (f"Option {answer} is correct because this cannot be the net of the given cube. "
                                    f"The positions of the {color_map.get(color0, color0)} and {color_map.get(color1, color1)} faces are swapped.")

    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_CubeUnfolding(folder_path, output_path, spacing=70, padding=100, font_size=60)
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    question = ("The left image shows a view of a colored cube. The options are nets that fold upwards to form a cube. "
                "Which net, when folded, CANNOT form the cube shown on the left? "
                "Please choose from options A, B, C, or D.")
    choices = ["A", "B", "C", "D"]
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
    
    
def prepare_composite_image_CubeUnfoldingComplex(folder_path):
    """
    Processes complex unfolded cube data to create a task question.

    Args:
        folder_path (str): The path to the folder with the data.
    """
    image_id = os.path.basename(folder_path)
    
    correct_view_files = [f for f in os.listdir(folder_path) if "_correct_" in f]
    incorrect_view_files = [f for f in os.listdir(folder_path) if "_incorrect_" in f]
    
    choices = ["A", "B", "C", "D"]
    random.shuffle(choices)
    answer = choices.pop(0)
    
    explanations = {}
    cube_file = [f for f in os.listdir(folder_path) if 'cube' in f][0]
    crop_blank_area(os.path.join(folder_path, cube_file))
    
    if random.choice([True, False]): # 1 correct, 3 incorrect
        correct_view_file = random.choice(correct_view_files)
        crop_blank_area(os.path.join(folder_path, correct_view_file))
        filename, ext = os.path.splitext(correct_view_file)
        os.rename(os.path.join(folder_path, correct_view_file), os.path.join(folder_path, f"{filename}_{answer}{ext}"))
        explanations[answer] = f"Option {answer} is correct because the relative positions of the three faces match the cube shown in the left image."
        
        selected_incorrect = random.sample(incorrect_view_files, 3)
        for incorrect_file in selected_incorrect:
            crop_blank_area(os.path.join(folder_path, incorrect_file))
            choice = choices.pop(0)
            filename, ext = os.path.splitext(incorrect_file)
            os.rename(os.path.join(folder_path, incorrect_file), os.path.join(folder_path, f"{filename}_{choice}{ext}"))
            if "change" in incorrect_file:
                explanations[choice] = "Option {choice} is incorrect because two faces have swapped positions."
            elif "transform" in incorrect_file:
                explanations[choice] = "Option {choice} is incorrect because the squares with asymmetric patterns have been rotated."
        
        question = ("The left image shows a cube with different patterns on its six faces from a particular viewing angle. "
                    "The options are nets of the cube, which are folded upward to form the cube. "
                    "Which net, when folded, can form the cube shown in the left image? Please choose from options A, B, C, or D.")
        
    else:  # 1 incorrect, 3 correct
        incorrect_file = random.choice(incorrect_view_files)
        crop_blank_area(os.path.join(folder_path, incorrect_file))
        filename, ext = os.path.splitext(incorrect_file)
        os.rename(os.path.join(folder_path, incorrect_file), os.path.join(folder_path, f"{filename}_{answer}{ext}"))
        if "change" in incorrect_file:
            explanations[answer] = "Option {answer} is correct because two faces have swapped positions, so it cannot form the cube shown in the left image."
        elif "transform" in incorrect_file:
            explanations[answer] = "Option {answer} is correct because the squares with asymmetric patterns have been rotated, so it cannot form the cube shown in the left image."
        
        selected_correct = random.sample(correct_view_files, 3)
        for correct_file in selected_correct:
            crop_blank_area(os.path.join(folder_path, correct_file))
            choice = choices.pop(0)
            filename, ext = os.path.splitext(correct_file)
            os.rename(os.path.join(folder_path, correct_file), os.path.join(folder_path, f"{filename}_{choice}{ext}"))
            explanations[choice] = "Option {choice} is incorrect because the relative positions of the three faces match the cube shown in the left image."
           
        question = ("The left image shows a cube with different patterns on its six faces from a particular viewing angle. "
                    "The options are nets of the cube, which are folded upward to form the cube. "
                    "Which net, when folded, CANNOT form the cube shown in the left image? Please choose from options A, B, C, or D.")
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_CubeUnfolding(folder_path, output_path, spacing=70, padding=100, font_size=60)
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    
    choices = ["A", "B", "C", "D"]
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
    

def prepare_composite_image_PaperFolding(folder_path):
    """
    Processes paper folding data to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
    """
    image_id = os.path.basename(folder_path)
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    ops = int(image_id.split('-')[3])
    create_composite_image_PaperFolding(folder_path, output_path, spacing=70, padding=100, font_size=60, ops=ops)
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    width, height = image_id.split('-')[1], image_id.split('-')[2]
    question = (f"The original paper is a {width}x{height} grid. "
                f"The top row of images shows the effect of folding the grid paper {ops} times. "
                "Folding operations include horizontal, vertical, and diagonal folds. "
                "The rightmost image in the top row shows the result after punching a hole. "
                "Which image shows the paper after it is unfolded? Black solid circles represent holes in the grid squares. "
                "Please choose from options A, B, C, or D.")
    choices = ["A", "B", "C", "D"]
    answer = [f for f in os.listdir(folder_path) if "correct" in f][0].split('_')[-1].split('.')[0]
    
    with open(os.path.join(folder_path, "info.json"), 'r', encoding="utf-8") as f:
        info = json.load(f)
    explanations = info["explanation"]
    
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)
