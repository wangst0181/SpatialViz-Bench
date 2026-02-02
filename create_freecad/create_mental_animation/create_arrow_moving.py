import random, json, os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import deque

class ArrowPath:
    """
    A class to generate and manipulate a path of an arrow on a 2D grid.
    """
    def __init__(self, x, y, k):
        """
        Initializes the ArrowPath object.

        Args:
            x (int): The width of the grid.
            y (int): The height of the grid.
            k (int): The number of steps in the path.
        """
        self.x = x  # grid width
        self.y = y  # grid height
        self.k = k  # number of steps
        self.max_step = min(x, y)  # maximum step length
        # Direction mapping: 0:Up, 1:Right, 2:Down, 3:Left
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # Initialize with a random starting position and direction
        self.pos = [random.randint(0, x-1), random.randint(0, y-1)]
        self.dir = random.randint(0, 3)
        self.path = []
        self.positions = [self.pos.copy()]
        self.orientations = [self.dir]

    def get_relative_directions(self):
        """
        Calculates relative directions (forward, backward, left, right) based on the current orientation.

        Returns:
            dict: A dictionary mapping relative direction names to their coordinate vectors.
        """
        forward = self.directions[self.dir]
        backward = (-forward[0], -forward[1])
        left_dir = (self.dir - 1) % 4
        right_dir = (self.dir + 1) % 4
        left = self.directions[left_dir]
        right = self.directions[right_dir]
        return {"forward": forward, "backward": backward, "left": left, "right": right}

    def is_valid_move(self, new_pos):
        """
        Checks if a new position is within the grid boundaries.

        Args:
            new_pos (list): The new [x, y] position.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        return 0 <= new_pos[0] < self.x and 0 <= new_pos[1] < self.y

    def move(self, rel_dir, steps):
        """
        Executes a single move in a relative direction for a given number of steps.

        Args:
            rel_dir (str): The relative direction ("forward", "backward", "left", "right").
            steps (int): The number of steps to move.

        Returns:
            bool: True if the move was successful, False if it was invalid.
        """
        rel_directions = self.get_relative_directions()
        move_dir = rel_directions[rel_dir]
        new_pos = [self.pos[0] + move_dir[0] * steps, self.pos[1] + move_dir[1] * steps]
        
        if not self.is_valid_move(new_pos):
            return False
        
        # Update position
        self.pos = new_pos
        self.positions.append(self.pos.copy())
        
        # Update orientation
        if rel_dir == "forward":
            pass  # Direction remains the same
        elif rel_dir == "backward":
            self.dir = (self.dir + 2) % 4  # Reverse direction
        elif rel_dir == "left":
            self.dir = (self.dir - 1) % 4  # 90 degrees counter-clockwise
        elif rel_dir == "right":
            self.dir = (self.dir + 1) % 4  # 90 degrees clockwise
        
        self.orientations.append(self.dir)
        self.path.append((rel_dir, steps))
        return True

    def generate_random_path(self):
        """
        Generates a random path of k steps, ensuring each move is valid.
        """
        for _ in range(self.k):
            rel_dir = random.choice(["forward", "backward", "left", "right"])
            steps = random.randint(1, self.max_step)
            # Ensure the move is valid
            while not self.move(rel_dir, steps):
                rel_dir = random.choice(["forward", "backward", "left", "right"])
                steps = random.randint(1, self.max_step)

    def generate_alternative_path(self, original_path, original_end, original_dir):
        """
        Generates an alternative path starting from the same point but ending at a different position or orientation.

        Args:
            original_path (list): The path of the correct solution.
            original_end (list): The final position of the correct path.
            original_dir (int): The final orientation of the correct path.

        Returns:
            bool: True if a valid alternative path was generated, False otherwise.
        """
        self.path = []
        self.positions = [self.pos.copy()]
        self.orientations = [self.dir]
        
        for _ in range(self.k):
            # Avoid generating a path identical to the original
            valid_move = False
            attempts = 0
            max_attempts = 10
            while not valid_move and attempts < max_attempts:
                rel_dir = random.choice(["forward", "backward", "left", "right"])
                steps = random.randint(1, self.max_step)
                # Temporarily save state
                temp_pos = self.pos.copy()
                temp_dir = self.dir
                # Try to move
                if self.move(rel_dir, steps):
                    valid_move = True
                    # If this is the final step, check if end position and orientation are different
                    if len(self.path) == self.k:
                        if (self.pos == original_end and self.dir == original_dir) or self.path == original_path:
                            valid_move = False
                            self.pos = temp_pos
                            self.dir = temp_dir
                            self.positions.pop()
                            self.orientations.pop()
                            self.path.pop()
                else:
                    valid_move = False
                attempts += 1
            if not valid_move:
                # If no valid move is found, restart
                self.pos = self.positions[0].copy()
                self.dir = self.orientations[0]
                self.path = []
                self.positions = [self.pos.copy()]
                self.orientations = [self.orientations[0]]
                return False
        return True

def plot_arrow_path(path_obj, save_path):
    """
    Visualizes the start and end points of a path on a grid with arrows and saves the plot.

    Args:
        path_obj (ArrowPath): The ArrowPath object to plot.
        save_path (str): The file path to save the plot to.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xticks(np.arange(0, path_obj.x, 1))
    ax.set_yticks(np.arange(0, path_obj.y, 1))
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    
    ax.set_xlim(0, path_obj.x)
    ax.set_ylim(0, path_obj.y)
    
    ax.set_aspect('equal', adjustable='box')
    
    arrow_length = 0.35
    head_width = 0.3
    head_length = 0.2
    width = 0.1
    
    total_length = arrow_length + head_length
    offset = (0.5 - total_length / 2)
    
    # Plot start arrow (red)
    start_pos = path_obj.positions[0]
    start_dir = path_obj.directions[path_obj.orientations[0]]
    start_x = start_pos[0] + 0.5 - start_dir[0] * offset
    start_y = start_pos[1] + 0.5 - start_dir[1] * offset
    ax.arrow(start_x, start_y, 
             start_dir[0] * arrow_length, start_dir[1] * arrow_length, 
             color='red', width=width, head_width=head_width, head_length=head_length)
    
    # Plot end arrow (green)
    end_pos = path_obj.positions[-1]
    end_dir = path_obj.directions[path_obj.orientations[-1]]
    end_x = end_pos[0] + 0.5 - end_dir[0] * offset
    end_y = end_pos[1] + 0.5 - end_dir[1] * offset
    ax.arrow(end_x, end_y, 
             end_dir[0] * arrow_length, end_dir[1] * arrow_length, 
             color='green', width=width, head_width=head_width, head_length=head_length)
    
    plt.title("Path: Start (Red) and End (Green) Arrows")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def dump_json(save_path, data):
    """
    Saves a dictionary to a JSON file.

    Args:
        save_path (str): The file path for the JSON file.
        data (dict): The dictionary to save.
    """
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main(args):
    """
    The main function to generate arrow moving puzzles and their corresponding data.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    x, y, k = args.grid_size[0], args.grid_size[1], args.steps
    save_folder = args.save_folder
    num_samples = args.num_samples
    
    num_directions = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

    i = 0
    while i < num_samples:
        save_name=f"{i}-{x}-{y}-{k}"
        os.makedirs(os.path.join(save_folder, save_name), exist_ok=True)
        json_data = {}
        
        # Generate the correct path
        path1 = ArrowPath(x, y, k)
        path1.generate_random_path()
        if path1.positions[0] == path1.positions[-1] and path1.orientations[0] == path1.orientations[-1]:
            continue
        
        print("Path 1:")
        correct_path = []
        for step, (rel_dir, steps) in enumerate(path1.path):
            correct_path.append((rel_dir, steps))
            print(f"Step {step + 1}: ({rel_dir}, {steps})")
        print(f"Start Position: {path1.positions[0]}, Start Direction: {path1.orientations[0]}")
        print(f"End Position: {path1.positions[-1]}, End Direction: {path1.orientations[-1]}")
        
        json_data["correct"] = {"Start_Position_Direction": (path1.positions[0], num_directions[path1.orientations[0]]), 
                                "End_Position_Direction": (path1.positions[-1], num_directions[path1.orientations[-1]]),
                                "Path": correct_path}
        
        # Visualize the correct path's start and end
        plot_arrow_path(path1, save_path=os.path.join(save_folder, save_name, f"{save_name}.png"))
        
        # Save info for alternative path generation
        path1_data = path1.path.copy()
        path1_end = path1.positions[-1].copy()
        path1_end_dir = path1.orientations[-1]
        start_pos = path1.positions[0].copy()
        start_dir = path1.orientations[0]
        
        # Generate three alternative paths
        alternative_paths = []
        path_num = 1
        
        while len(alternative_paths) < 3:
            incorrect_path = []
            new_path = ArrowPath(x, y, k)
            new_path.pos = start_pos.copy()
            new_path.dir = start_dir
            if new_path.generate_alternative_path(path1_data, path1_end, path1_end_dir):
                alternative_paths.append(new_path)
                print(f"\nAlternative Path {path_num}:")
                for step, (rel_dir, steps) in enumerate(new_path.path):
                    incorrect_path.append((rel_dir, steps))
                    print(f"Step {step + 1}: ({rel_dir}, {steps})")
                print(f"Start Position: {new_path.positions[0]}, Start Direction: {new_path.orientations[0]}")
                print(f"End Position: {new_path.positions[-1]}, End Direction: {new_path.orientations[-1]}")
                
                json_data[f"incorrect{path_num}"] = {"Start_Position_Direction": (new_path.positions[0], num_directions[new_path.orientations[0]]), 
                                                     "End_Position_Direction": (new_path.positions[-1], num_directions[new_path.orientations[-1]]),
                                                     "Path": incorrect_path}
                
                path_num += 1
            
        print(json_data)
        dump_json(os.path.join(save_folder, save_name, "info.json"), json_data)
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arrow path puzzles for mental animation benchmark.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to the directory to save the generated data.")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[5, 5], help="The size of the grid (width height).")
    parser.add_argument("--steps", type=int, default=3, help="The number of steps in the path.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of data samples to generate.")

    args = parser.parse_args()
    main(args)
    

"""
    python create_arrow_moving.py 
        --save_folder "/path/to/save/your/data" 
        --grid_size 5 5 
        --steps 3 
        --num_samples 40
"""