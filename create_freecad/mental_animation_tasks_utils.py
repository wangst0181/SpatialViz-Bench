import random, os, json, shutil, time
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import *

def create_composite_image_ArrowMoving(folder_path, output_path, spacing=70, padding=100, font_size=60, mode=0):
    """
    Creates a composite reference image for Arrow Moving problems.

    Args:
        folder_path (str): The path to the folder containing the source images.
        output_path (str): The path to save the composite image.
        spacing (int, optional): The spacing between images. Defaults to 70.
        padding (int, optional): The padding around the canvas. Defaults to 100.
        font_size (int, optional): The font size for the labels. Defaults to 60.
        mode (int, optional): The display mode (0 for start/end, 1 for start and options). Defaults to 0.

    Raises:
        FileNotFoundError: If a required image is not found in the folder.
    """
    
    # Read all PNG images
    if mode == 0 :
        images = {}
        filenames = {"start": "", "end": ""}
        for file in os.listdir(folder_path):
            if "path1" in file:
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
        
        upper_width = widths["start"] + widths["end"] + spacing
        upper_height = max(heights["start"], heights["end"])
        
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
        canvas.paste(images["start"], (upper_x, upper_y))
        canvas.paste(images["end"], (upper_x + widths["start"] + spacing, upper_y))
        
        # Add text above each image and ensure the font is visible
        text_positions = [
            (upper_x + widths["start"] // 2, upper_y - font_size - 10, "Initial State"),
            (upper_x + widths["start"] + spacing + widths["end"] // 2, upper_y - font_size - 10, "Final State"),
        ]
                
    elif mode == 1:
        images = {}
        filenames = {"start": "", "A": "", "B": "", "C": "", "D": ""}
        for file in os.listdir(folder_path):
            if file.endswith('.png'):
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
        upper_width = widths["start"] + widths["A"] + widths["B"] + widths["C"] + widths["D"] + 4 * spacing
        upper_height = max(heights["start"], heights["A"], heights["B"], heights["C"], heights["D"])
        
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
        canvas.paste(images["start"], (upper_x, upper_y))
        canvas.paste(images["A"], (upper_x + widths["start"] + spacing, upper_y))
        canvas.paste(images["B"], (upper_x + widths["start"] + widths["A"] + 2*spacing, upper_y))
        canvas.paste(images["C"], (upper_x + widths["start"] + widths["A"] + widths["B"] + 3*spacing, upper_y))
        canvas.paste(images["D"], (upper_x + widths["start"] + widths["A"] + widths["B"] + widths["C"] + 4*spacing, upper_y))
        
        # Add text above each image and ensure the font is visible
        text_positions = [
            (upper_x + widths["start"] // 2, upper_y - font_size - 10, "Initial State"),
            (upper_x + widths["start"] + spacing + widths["A"] // 2, upper_y - font_size - 10, "A"),
            (upper_x + widths["start"] + widths["A"] + 2 * spacing + widths["B"] // 2, upper_y - font_size - 10, "B"),
            (upper_x + widths["start"] + widths["A"] + widths["B"] + 3 * spacing + widths["C"] // 2, upper_y - font_size - 10, "C"),
            (upper_x + widths["start"] + widths["A"] + widths["B"] + widths["C"] + 4 * spacing + widths["D"] // 2, upper_y - font_size - 10, "D")
        ]

    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")


def create_composite_image_BlocksMoving(folder_path, output_path, spacing=70, padding=100, font_size=60):
    """
    Creates a composite reference image for block moving problems.

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
    filenames = {"init": "", "init_back": "", "transformed": "", "transformed_back": ""}
    for file in os.listdir(folder_path):
        if file.endswith('.png'):
            filenames[file.split('.')[0]] = file
            
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
    upper_width = widths["init"] + widths["init_back"] + spacing
    lower_width = widths["transformed"] + widths["transformed_back"] + spacing
    max_total_width = max(upper_width, lower_width)
    
    upper_height = max(heights["init"], heights["init_back"])
    lower_height = max(heights["transformed"], heights["transformed_back"])
    
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
    canvas.paste(images["init"], (upper_x, upper_y))
    canvas.paste(images["init_back"], (upper_x + widths["init"] + spacing, upper_y))
    
    # Calculate the starting x position for the lower row
    lower_x = (total_width - lower_width) // 2
    lower_y = upper_y + upper_height + padding + font_size
    
    # Place the lower row images
    canvas.paste(images["transformed"], (lower_x, lower_y))
    canvas.paste(images["transformed_back"], (lower_x + widths["transformed"] + spacing, lower_y))
    
    # Add text above each image and ensure the font is visible
    text_positions = [
        (upper_x + widths["init"] // 2, upper_y - font_size - 10, "Initial State"),
        (upper_x + widths["init"] + spacing + widths["init_back"] // 2, upper_y - font_size - 10, "Rotated Initial State"),
        (lower_x + widths["transformed"] // 2, lower_y - font_size - 10, "​Final State"),
        (lower_x + widths["transformed"] + spacing + widths["transformed_back"] // 2, lower_y - font_size - 10, "​Rotated Final State"),
    ]
    
    for x, y, text in text_positions:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y), text, fill="black", font=font)
    
    # Save the output image
    canvas.save(output_path, "PNG")
    print(f"Composite image saved at: {output_path}")
    

def prepare_composite_image_BlocksMoving(folder_path):
    """
    Processes moving blocks data to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
    """
    image_id = os.path.basename(folder_path)
    png_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
    for pth in png_paths:
        crop_blank_area(pth)
    
    output_path = os.path.join(folder_path, f"{image_id}.png")
    create_composite_image_BlocksMoving(folder_path, output_path, spacing=70, padding=100, font_size=60)
    
    with open(os.path.join(folder_path, "info.json"), 'r', encoding="utf-8") as f:
        info = json.load(f)
    
    directions = {(1, 0, 0): "positive x", (-1, 0, 0): "negative x", (0, 1, 0): "positive y", 
                  (0, -1, 0): "negative y", (0, 0, 1): "positive z", (0, 0, -1): "negative z"}
    correct_transformation = info["correct"]
    incorrect_transformations = info["incorrect"]
    
    choices = []
    
    def format_transformation(transformation):
        steps = []
        for obj, direction in transformation:
            direction_text = directions.get(tuple(direction), "unknown")
            steps.append(f"{tuple(obj)} {direction_text}")
        return ' --> '.join(steps)

    correct_choice = format_transformation(correct_transformation)
    choices.append(correct_choice)
    
    for transformation in incorrect_transformations:
        choices.append(format_transformation(transformation))
    
    random.shuffle(choices)
    answer_id = choices.index(correct_choice)
    option_letters = ["A", "B", "C", "D"]
    answer = option_letters[answer_id]
    
    remaining_choices = "".join([opt for i, opt in enumerate(option_letters) if i != answer_id])
    explanations = {
        answer: "Option {answer} is correct because this sequence of transformations correctly changes the initial state to the final state.",
        remaining_choices: f"Options {remaining_choices} are incorrect because these transformations do not result in the final state."
    }
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    question = ("The top row shows different views of a cube stack in its initial state. The bottom row shows views of the final state after a transformation. "
                "During the transformation, a block can move one unit forward, backward, left, right, up, or down. "
                "If the target position is empty, the block moves there. If the target position is occupied, the blocks swap places. "
                "Blocks cannot float in mid-air. If a block moves to an empty space, any blocks above it will fall until they are supported. "
                "The xyz axes are as shown. A block's position can be specified by coordinates (x,y,z). "
                "Which of the following transformation sequences can change the cube stack from the initial to the final state? "
                "Please choose from options A, B, C, or D.")
    save_json(category, question_type, level, image_id, question, choices, answer, explanations, folder_path)


def prepare_composite_image_ArrowMoving(folder_path):
    """
    Processes arrow moving data to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
    """
    image_id = os.path.basename(folder_path)
    
    with open(os.path.join(folder_path, "info.json"), 'r', encoding="utf-8") as f:
        info = json.load(f)

    correct_path = info["correct"]["Path"]
    incorrect_paths = [info[f"incorrect{i}"]["Path"] for i in range(1, 4)]
    
    direction_map = {"backward": "Backward", "forward": "Forward", "left": "Left", "right": "Right"}
    
    def format_path(path):
        steps = []
        for rel_dir, num_steps in path:
            steps.append(f"({direction_map.get(rel_dir, rel_dir)}, {num_steps} units)")
        return ' --> '.join(steps)

    choices_list = [format_path(p) for p in incorrect_paths]
    correct_path_text = format_path(correct_path)
    
    choices = ["A", "B", "C", "D"]
    random.shuffle(choices)
    answer = choices[0]
    
    choices_dict = {answer: correct_path_text}
    for i, incorrect_text in enumerate(choices_list):
        choices_dict[choices[i+1]] = incorrect_text
        
    choices_text = [choices_dict[key] for key in sorted(choices_dict)]
        
    remaining_choices = "".join(sorted([c for c in choices if c != answer]))
    explanations = {
        answer: "Option {answer} is correct because the initial arrow can reach the final state via this transformation.",
        remaining_choices: f"Options {remaining_choices} are incorrect because the initial arrow cannot reach the final state via these transformations."
    }
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]
    question = ("The red arrow is the initial arrow, and the green arrow is the final arrow. "
                "The arrow can move forward, backward, left, or right on the map, relative to its own orientation. "
                "The direction the arrow points is 'forward'. After moving, the arrow's orientation updates to the direction of movement. "
                "Which of the following paths allows the arrow to get from the start to the end position? "
                "Please choose from options A, B, C, or D.")
    save_json(category, question_type, level, image_id, question, choices_text, answer, explanations, folder_path)
    
    
def prepare_composite_image_ArrowMoving_complex(folder_path, mode=0):
    """
    Processes complex arrow moving data to create a task question.

    Args:
        folder_path (str): The path to the folder containing the data.
        mode (int): The mode for the task. 0 for path selection, 1 for final state selection.
    """
    image_id = os.path.basename(folder_path)
    
    with open(os.path.join(folder_path, "info.json"), 'r', encoding="utf-8") as f:
        info = json.load(f)
    
    correct_path = info["correct"]
    incorrect_paths = [info[f"incorrect{i}"] for i in range(1, 4)]
    
    direction_map = {"backward": "Backward", "forward": "Forward", "left": "Left", "right": "Right"}
    
    def format_path(path):
        steps = []
        for (x, y), direction, num_steps in path:
            steps.append(f"(at ({x},{y}) move {direction_map.get(direction, direction)}, {num_steps} units)")
        return ' --> '.join(steps)

    choices = ["A", "B", "C", "D"]
    random.shuffle(choices)
    answer = choices[0]
    remaining_choices_str = "".join(sorted([c for c in choices if c != answer]))
    
    category, question_type, level = folder_path.split(os.sep)[-4:-1]

    if mode == 0:
        correct_path_text = format_path(correct_path)
        incorrect_path_texts = [format_path(p) for p in incorrect_paths]
        
        choices_dict = {answer: correct_path_text}
        for i, text in enumerate(incorrect_path_texts):
            choices_dict[choices[i+1]] = text
            
        choices_text = [choices_dict[key] for key in sorted(choices_dict)]
            
        output_path = os.path.join(folder_path, f"{image_id}.png")
        create_composite_image_ArrowMoving(folder_path, output_path, spacing=70, padding=100, font_size=60, mode=0)
            
        explanations = {
            answer: "Option {answer} is correct because this path transforms the initial state to the final state.",
            remaining_choices_str: f"Options {remaining_choices_str} are incorrect because these paths do not lead to the final state."
        }
        question = ("The left image is the initial state, and the right image is the final state. "
                    "Arrows can move forward, backward, left, or right relative to their own orientation. "
                    "After moving, an arrow's orientation updates to the direction of movement. "
                    "If the target position is empty, the arrow moves there. Otherwise, it swaps with the arrow at the target position, with both moves following the rules. "
                    "Which of the following paths transforms the grid from the initial to the final state? "
                    "Please choose from options A, B, C, or D.")
        save_json(category, question_type, level, image_id, question, choices_text, answer, explanations, folder_path)
    elif mode == 1:
        correct_path_text = format_path(correct_path)
        
        correct_end_file = "path1_end.png"
        incorrect_end_files = [f for f in os.listdir(folder_path) if "incorrect" in f]
        
        os.rename(os.path.join(folder_path, correct_end_file), os.path.join(folder_path, f"path1_end_{answer}.png"))
        
        for i, incorrect_file in enumerate(incorrect_end_files):
            filename, ext = os.path.splitext(incorrect_file)
            os.rename(os.path.join(folder_path, incorrect_file), os.path.join(folder_path, f"{filename}_{choices[i+1]}{ext}"))
        
        output_path = os.path.join(folder_path, f"{image_id}.png")
        create_composite_image_ArrowMoving(folder_path, output_path, spacing=70, padding=100, font_size=60, mode=1)
        
        choices_options = ["A", "B", "C", "D"]
        explanations = {
            answer: "Option {answer} is correct because the initial state can reach this state after the transformation.",
            remaining_choices_str: f"Options {remaining_choices_str} are incorrect because the initial state cannot reach these states after the transformation."
        }
        question = (f"The left image is the initial state. Arrows move according to the rules described (relative movement, orientation change, swapping). "
                    f"Which of the options shows the final state after the transformation: {correct_path_text}? "
                    "Please choose from options A, B, C, or D.")
        save_json(category, question_type, level, image_id, question, choices_options, answer, explanations, folder_path)