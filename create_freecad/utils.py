import random, os, json, shutil, time
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def detect_subfigures(pdf_path, save_folder, padding=10):
    """
    Automatically detects subfigure regions in a PDF page and saves them as separate PNG images.
    Args:
        pdf_path (str): The path to the input PDF file.
        save_folder (str): The folder where the output PNG images will be saved.
        padding (int, optional): The padding to add around the detected subfigures. Defaults to 10.
    Returns:
        Bool: True if subfigures are detected and saved, otherwise False.
    """
    # Open the PDF file
    doc = fitz.open(pdf_path)
    page = doc[0]  # Select the first page

    # Render the PDF page as an image with higher resolution
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use edge detection to find subfigure regions
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[:3]
    
    # Get bounding boxes and sort them
    rects = [cv2.boundingRect(c) for c in contours]
    sorted_by_y = sorted(rects, key=lambda r: r[1])  # Sort by Y coordinate

    # Separate the upper and lower row subfigures
    if len(sorted_by_y) < 3:
        return False
    upper_rects = sorted(sorted_by_y[:2], key=lambda r: r[0])  # Sort upper row by X coordinate
    front_rect = upper_rects[0]  # Top-left
    left_rect = upper_rects[1]   # Top-right
    top_rect = sorted_by_y[2]    # Bottom-left

    # Define subfigure names and their corresponding regions
    subfigures = [
        (front_rect, "Front"),
        (left_rect, "Left"),
        (top_rect, "Top")
    ]
    
    # Crop and save subfigures
    for rect, name in subfigures:
        x, y, w, h = rect
        # Add white padding
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, img.shape[1] - x)
        h = min(h + 2 * padding, img.shape[0] - y)

        sub_img = img[y:y+h, x:x+w]
        output_path = f"{save_folder}/{name}.png"
        cv2.imwrite(output_path, cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR))
        print(f"Subfigure {name} saved as {output_path}")
    
    doc.close()
    return True

def crop_blank_area(png_path, padding=10):
    """
    Removes the white border from a PNG image and saves the cropped image.
    Args:
        png_path (str): The path to the PNG image.
        padding (int, optional): Padding to add around the cropped content. Defaults to 10.
    """
    img = np.array(Image.open(png_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_x = float('inf')
    min_y = float('inf')
    max_right = 0  # Maximum right boundary (x + w)
    max_bottom = 0  # Maximum bottom boundary (y + h)

    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_right = max(max_right, x + w)
        max_bottom = max(max_bottom, y + h)

    min_x = max(min_x - padding, 0)
    min_y = max(min_y - padding, 0)
    max_right = min(max_right + padding, img.shape[1])
    max_bottom = min(max_bottom + padding, img.shape[0])
    sub_img = img[min_y:max_bottom, min_x:max_right]
    
    cv2.imwrite(png_path, cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR))
    print(f"Cropped image saved at: {png_path}")


def check_symmetric(image, threshold=1):
    """
    Checks if an image is symmetric horizontally, vertically, or both.
    Args:
        image (np.array): The input image as a NumPy array.
        threshold (int, optional): The tolerance for symmetry checking. Defaults to 1.
    Returns:
        str: 'both', 'vertical', 'horizontal', or 'unsymmetric'.
    """
    if np.mean((image - cv2.flip(image, 1))**2) <= threshold and np.mean((image - cv2.flip(image, 0))**2) <= threshold:
        return 'both'
    if np.mean((image - cv2.flip(image, 1))**2) <= threshold:
        return 'vertical'
    if np.mean((image - cv2.flip(image, 0))**2) <= threshold:
        return 'horizontal'
    return 'unsymmetric'


def find_max_area(folder_path):
    """
    Finds the view with the largest area in a folder and renames it as the reference view.
    Args:
        folder_path (str): Path to the folder containing 'Front.png', 'Left.png', 'Top.png'.
    Returns:
        str: The name of the reference view ('Front', 'Left', or 'Top').
    """
    images_name = ["Front", "Left", "Top"]
    max_area = -1
    ref = None
    for image_name in images_name:
        image = cv2.imread(f"{folder_path}/{image_name}.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Binarize the image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find external contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate the total area of all external contours
        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            total_area += area 
        
        if total_area > max_area:
            max_area = total_area
            ref = image_name
    os.rename(f"{folder_path}/{ref}.png", f"{folder_path}/{ref}_ref.png")
    return ref

def resize_image_keep_aspect_pil(img, target_height):
    """
    Resizes a PIL image to a target height while maintaining the aspect ratio.
    Args:
        img (PIL.Image.Image): The input PIL image.
        target_height (int): The desired height of the resized image.
    Returns:
        PIL.Image.Image: The resized image.
    """
    w, h = img.size
    scale = target_height / h
    new_w = int(w * scale)
    resized_img = img.resize((new_w, Image.LANCZOS))
    return resized_img


def rotate_and_flip_figure(image_path):
    """
    Generates rotated and flipped versions of an image for creating incorrect options.
    Args:
        image_path (str): The path to the input image.
    Returns:
        tuple: A tuple containing the rotation degrees for the correct rotation, horizontal flip, and vertical flip.
    """
    image = cv2.imread(image_path)
    os.rename(image_path, f"{image_path.split('.')[0]}_original.png")
    
    Rotation_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    Rotation_180 = cv2.rotate(image, cv2.ROTATE_180)
    Rotation_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    Rotation_images = [Rotation_90, Rotation_180, Rotation_270]
    rotations = ["90", "180", "270"]
    Rotation_id = random.choice([0,1,2])
    
    cv2.imwrite(f"{image_path.split('.')[0]}_{rotations[Rotation_id]}_rotated.png", Rotation_images[Rotation_id])
    
    h_id = random.choice([0,1,2])
    v_id = random.choice([0,1,2])
    flipped_horizontal = cv2.flip(Rotation_images[h_id], 0) # Flip vertically (upside down)
    flipped_vertical = cv2.flip(Rotation_images[v_id], 1) # Flip horizontally (left-right)
    cv2.imwrite(f"{image_path.split('.')[0]}_{rotations[h_id]}_horizontal.png", flipped_horizontal)
    cv2.imwrite(f"{image_path.split('.')[0]}_{rotations[v_id]}_vertical.png", flipped_vertical)
    return rotations[Rotation_id], rotations[h_id], rotations[v_id]


def save_json(category, question_type, level, image_id, question, choices, answer, explanation, folder_path):
    """
    Saves the question and related data to a JSON file.
    Args:
        category (str): The category of the question.
        question_type (str): The type of the question/task.
        level (str): The difficulty level.
        image_id (str): The unique ID for the image.
        question (str): The question text.
        choices (list): A list of choices for the question.
        answer (str): The correct answer.
        explanation (dict): A dictionary containing explanations.
        folder_path (str): The path to the folder where the JSON file will be saved.
    """
    json_data = [
        {
            "Category": category,
            "Task": question_type,
            "Level": level,
            "Image_id": image_id,
            "Question": question,
            "Choices": choices,
            "Answer": answer,
            "Explanation": explanation
        }
    ]
    
    data_file = f"{folder_path}/data.json"
    # If you want to append to an existing file, uncomment the following lines.
    # if os.path.exists(data_file):
    #     with open(data_file, "r", encoding="utf-8") as file:
    #         data = json.load(file)
    #     json_data += data
        
    with open(data_file, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)