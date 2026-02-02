import random, json, os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import deque

class ArrowMap:
    """
    A class to generate and manipulate a map of arrows and simulate their movements.
    """
    def __init__(self, x, y, k, colors=["red", "blue", "green", "purple", "yellow", "pink"]):
        """
        Initializes the ArrowMap object.

        Args:
            x (int): The width of the grid (x-axis, columns).
            y (int): The height of the grid (y-axis, rows).
            k (int): The number of steps in the path.
            colors (list): A list of possible colors for the arrows.
        """
        self.x = x  # map width (columns)
        self.y = y  # map height (rows)
        self.k = k  # number of steps
        self.max_step = min(x, y)  # maximum step length
        self.colors = colors
        # Direction mapping: Up, Right, Down, Left (Cartesian: (dx, dy))
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left
        self.map = self.initialize_map()
        self.path = []
        self.states = []

    def initialize_map(self):
        """
        Randomly initializes the map grid with arrows. Each cell has a 50% chance of containing an arrow.

        Returns:
            list: A 2D list representing the initialized map grid.
        """
        map_grid = [[None for _ in range(self.x)] for _ in range(self.y)]
        for i in range(self.y):
            for j in range(self.x):
                if random.random() < 0.5:
                    color = random.choice(self.colors)
                    direction = random.randint(0, 3)
                    map_grid[i][j] = {"color": color, "direction": direction}
        return map_grid

    def get_relative_directions(self, direction):
        """
        Calculates relative directions (forward, backward, left, right) based on the given absolute direction.

        Args:
            direction (int): The absolute direction (0-3).

        Returns:
            dict: A dictionary mapping relative direction names to their coordinate vectors.
        """
        forward = self.directions[direction]
        backward = (-forward[0], -forward[1])
        left_dir = (direction - 1) % 4
        right_dir = (direction + 1) % 4
        left = self.directions[left_dir]
        right = self.directions[right_dir]
        return {"forward": forward, "backward": backward, "left": left, "right": right}

    def is_valid_position(self, pos):
        """
        Checks if a given position is within the map boundaries.

        Args:
            pos (list): The [row, column] position.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        return 0 <= pos[0] < self.y and 0 <= pos[1] < self.x

    def move(self, arrow_pos, rel_dir, steps):
        """
        Performs a single move of an arrow, updating the map state and path.

        Args:
            arrow_pos (list): The [row, column] of the arrow to move.
            rel_dir (str): The relative direction of the move ("forward", "backward", "left", "right").
            steps (int): The number of steps to move.

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        if not self.is_valid_position(arrow_pos) or self.map[arrow_pos[0]][arrow_pos[1]] is None:
            return False

        curr_pos = arrow_pos
        curr_dir = self.map[curr_pos[0]][curr_pos[1]]["direction"]
        curr_color = self.map[curr_pos[0]][curr_pos[1]]["color"]

        rel_directions = self.get_relative_directions(curr_dir)
        move_dir = rel_directions[rel_dir]
        
        new_pos = [curr_pos[0] - move_dir[1] * steps, curr_pos[1] + move_dir[0] * steps]
        if not self.is_valid_position(new_pos):
            return False

        new_dir = curr_dir
        if rel_dir == "backward":
            new_dir = (curr_dir + 2) % 4
        elif rel_dir == "left":
            new_dir = (curr_dir - 1) % 4
        elif rel_dir == "right":
            new_dir = (curr_dir + 1) % 4
            
        if new_pos == curr_pos and new_dir == curr_dir:
            return False

        self.save_map_state()

        actual_steps = abs(new_pos[0] - curr_pos[0]) + abs(new_pos[1] - curr_pos[1])
        
        # Path format: ((x_col, y_row), rel_dir, steps)
        self.path.append(((curr_pos[1], self.y-1-curr_pos[0]), rel_dir, actual_steps))

        if self.map[new_pos[0]][new_pos[1]] is None:
            self.map[new_pos[0]][new_pos[1]] = {"color": curr_color, "direction": new_dir}
            self.map[curr_pos[0]][curr_pos[1]] = None
        else:
            target_arrow = self.map[new_pos[0]][new_pos[1]].copy()
            target_dir = target_arrow["direction"]
            target_move_dir = (-move_dir[0], -move_dir[1])
            
            rel_dirs = self.get_relative_directions(target_dir)
            
            rel_dir_key = next((key for key, value in rel_dirs.items() if value == target_move_dir), None)
            
            new_target_dir = target_dir
            if rel_dir_key == "backward":
                new_target_dir = (target_dir + 2) % 4
            elif rel_dir_key == "left":
                new_target_dir = (target_dir - 1) % 4
            elif rel_dir_key == "right":
                new_target_dir = (target_dir + 1) % 4

            self.map[new_pos[0]][new_pos[1]] = {"color": curr_color, "direction": new_dir}
            self.map[curr_pos[0]][curr_pos[1]] = {
                "color": target_arrow["color"],
                "direction": new_target_dir,
            }

        return True

    def save_map_state(self):
        """
        Saves a deep copy of the current map state to the state history.
        """
        map_copy = [
            [
                self.map[i][j].copy() if self.map[i][j] is not None else None
                for j in range(self.x)
            ]
            for i in range(self.y)
        ]
        self.states.append(map_copy)

    def generate_random_path(self):
        """
        Generates a random path of k steps, selecting a random arrow for each move.

        Returns:
            bool: True if a valid path of k steps was generated, False otherwise.
        """
        self.save_map_state()

        for _ in range(self.k):
            possible_starts = [
                [i, j]
                for i in range(self.y)
                for j in range(self.x)
                if self.map[i][j] is not None
            ]
            if not possible_starts:
                return False

            arrow_pos = random.choice(possible_starts)
            rel_dir = random.choice(["forward", "backward", "left", "right"])
            steps = random.randint(1, self.max_step)
            attempts = 0
            max_attempts = 10
            while not self.move(arrow_pos, rel_dir, steps) and attempts < max_attempts:
                arrow_pos = random.choice(possible_starts)
                rel_dir = random.choice(["forward", "backward", "left", "right"])
                steps = random.randint(1, self.max_step)
                attempts += 1
            if attempts >= max_attempts:
                return False

        return True

    def generate_alternative_path(self, original_end_map):
        """
        Generates an alternative path with the same starting map but a different end state.

        Args:
            original_end_map (list): The final map state of the correct solution.

        Returns:
            bool: True if a valid alternative path with a different end state was generated, False otherwise.
        """
        # Reset map to initial state
        self.path = []
        self.states = [self.states[0]]

        for _ in range(self.k):
            possible_starts = [
                [i, j]
                for i in range(self.y)
                for j in range(self.x)
                if self.map[i][j] is not None
            ]
            if not possible_starts:
                return False

            arrow_pos = random.choice(possible_starts)
            rel_dir = random.choice(["forward", "backward", "left", "right"])
            steps = random.randint(1, self.max_step)
            attempts = 0
            max_attempts = 10
            while not self.move(arrow_pos, rel_dir, steps) and attempts < max_attempts:
                arrow_pos = random.choice(possible_starts)
                rel_dir = random.choice(["forward", "backward", "left", "right"])
                steps = random.randint(1, self.max_step)
                attempts += 1
            if attempts >= max_attempts:
                return False

        # Check if the final map state is the same as the original
        if self.map == original_end_map:
            return False
        return True

def plot_map(arrow_map, map_grid, save_path, title):
    """
    Plots the arrow map grid and saves it to a file.

    Args:
        arrow_map (ArrowMap): The ArrowMap object containing grid dimensions.
        map_grid (list): The 2D map grid to plot.
        save_path (str): The file path to save the plot.
        title (str): The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(0, arrow_map.x, 1))
    ax.set_yticks(np.arange(0, arrow_map.y, 1))
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.set_xlim(0, arrow_map.x)
    ax.set_ylim(0, arrow_map.y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    arrow_length = 0.35
    head_width = 0.3
    head_length = 0.2
    width = 0.1
    total_length = arrow_length + head_length
    offset = (0.5 - total_length / 2)

    for i in range(arrow_map.y):
        for j in range(arrow_map.x):
            if map_grid[i][j] is not None:
                color = map_grid[i][j]["color"]
                direction = arrow_map.directions[map_grid[i][j]["direction"]]
                
                plot_x = j
                plot_y = arrow_map.y - 1 - i
                start_x = plot_x + 0.5 - direction[0] * offset
                start_y = plot_y + 0.5 - direction[1] * offset
                ax.arrow(
                    start_x,
                    start_y,
                    direction[0] * arrow_length,
                    direction[1] * arrow_length,
                    color=color,
                    width=width,
                    head_width=head_width,
                    head_length=head_length,
                )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def dump_json(save_path, data):
    """
    Saves a dictionary to a JSON file, creating the directory if it doesn't exist.

    Args:
        save_path (str): The file path for the JSON file.
        data (dict): The dictionary to save.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main(args):
    """
    The main function to generate arrow map puzzles and their corresponding data.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    cols, rows, k = args.grid_size[0], args.grid_size[1], args.steps
    save_folder = args.save_folder
    num_samples = args.num_samples
    
    i = 0
    while i < num_samples:
        save_name=f"{i}-{cols}-{rows}-{k}"
        
        json_data = {}

        path1 = ArrowMap(cols, rows, k)
        if not path1.generate_random_path():
            print("Could not generate a valid path, retrying...")
            continue

        print(f"\n--- Generating sample {i} ---")
        print("Correct Path:")
        correct_path = []
        for step, ((x, y), rel_dir, steps) in enumerate(path1.path):
            correct_path.append(((x, y), rel_dir, steps))
            print(f"Step {step + 1}: (({x}, {y}), {rel_dir}, {steps})")

        json_data["correct"] = correct_path

        plot_map(path1, path1.states[0], os.path.join(save_folder, save_name, "path1_start.png"), "Start State")
        plot_map(path1, path1.map, os.path.join(save_folder, save_name, "path1_end.png"), "End State")

        path1_end_map = [
            [
                path1.map[i][j].copy() if path1.map[i][j] is not None else None
                for j in range(path1.x)
            ]
            for i in range(path1.y)
        ]

        alternative_paths = []
        path_num = 1

        while len(alternative_paths) < 3:
            new_path = ArrowMap(cols, rows, k)
            new_path.map = [
                [
                    path1.states[0][i][j].copy() if path1.states[0][i][j] is not None else None
                    for j in range(path1.x)
                ]
                for i in range(path1.y)
            ]
            
            new_path.states = [new_path.map]
            if new_path.generate_alternative_path(path1_end_map):
                alternative_paths.append(new_path)
                print(f"\nAlternative Path {path_num}:")
                incorrect_path = []
                for step, ((x, y), rel_dir, steps) in enumerate(new_path.path):
                    incorrect_path.append(((x, y), rel_dir, steps))
                    print(f"Step {step + 1}: (({x}, {y}), {rel_dir}, {steps})")

                json_data[f"incorrect{path_num}"] = incorrect_path
                plot_map(
                    new_path,
                    new_path.map,
                    os.path.join(save_folder, save_name, f"path_incorrect{path_num}_end.png"),
                    f"Alternative Path {path_num} End State",
                )
                path_num += 1

        dump_json(os.path.join(save_folder, save_name, "info.json"), json_data)
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arrow map puzzles for mental animation benchmark.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to the directory to save the generated data.")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[5, 5], help="The size of the grid (width height).")
    parser.add_argument("--steps", type=int, default=3, help="The number of steps in the path.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of data samples to generate.")

    args = parser.parse_args()
    main(args)
    
"""
    python your_script_name.py 
    --save_folder "/path/to/save/your/data" 
    --grid_size 3 3 
    --steps 4 
    --num_samples 37
"""