import os
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import json
import cv2

class Paper:
    def __init__(self, rows, cols):
        """
        Initializes the paper grid. 'rows' and 'cols' are the dimensions.
        All positions are initially 0 (no holes).

        Args:
            rows (int): The number of rows in the paper grid.
            cols (int): The number of columns in the paper grid.
        """
        self.original_rows = rows  # Original number of rows
        self.original_cols = cols  # Original number of columns
        self.grid = np.zeros((rows, cols))  # Current grid state
        self.complete_grid = np.zeros((rows, cols))  # Complete grid state
        self.current_rows = rows  # Current number of rows (may change after folding)
        self.current_cols = cols  # Current number of columns (may change after folding)
        self.folds = []  # Records fold operations for unfolding
        self.visited = np.zeros((rows, cols), dtype=bool)

    def get_start_row(self, matrix=None):
        """
        Finds the first row index of the current active paper region.

        Args:
            matrix (np.ndarray, optional): The matrix to search within. Defaults to None,
                                           in which case `self.complete_grid` is used.

        Returns:
            int: The starting row index.
        """
        start_row = 0
        if matrix is None:
            matrix = self.complete_grid
        for i in range(self.original_rows):
            if all(matrix[i][j] == -1 for j in range(self.original_cols)):
                start_row = i + 1
            else:
                break
        return start_row

    def get_end_row(self, matrix=None):
        """
        Finds the last row index of the current active paper region.

        Args:
            matrix (np.ndarray, optional): The matrix to search within. Defaults to None,
                                           in which case `self.complete_grid` is used.

        Returns:
            int: The ending row index.
        """
        end_row = self.original_rows - 1
        if matrix is None:
            matrix = self.complete_grid
        for i in range(self.original_rows - 1, -1, -1):
            if all(matrix[i][j] == -1 for j in range(self.original_cols)):
                end_row = i - 1
            else:
                break
        return end_row

    def get_start_col(self, matrix=None):
        """
        Finds the first column index of the current active paper region.

        Args:
            matrix (np.ndarray, optional): The matrix to search within. Defaults to None,
                                           in which case `self.complete_grid` is used.

        Returns:
            int: The starting column index.
        """
        start_col = 0
        if matrix is None:
            matrix = self.complete_grid
        for j in range(self.original_cols):
            if all(matrix[i][j] == -1 for i in range(self.original_rows)):
                start_col = j + 1
            else:
                break
        return start_col

    def get_end_col(self, matrix=None):
        """
        Finds the last column index of the current active paper region.

        Args:
            matrix (np.ndarray, optional): The matrix to search within. Defaults to None,
                                           in which case `self.complete_grid` is used.

        Returns:
            int: The ending column index.
        """
        end_col = self.original_cols - 1
        if matrix is None:
            matrix = self.complete_grid
        for j in range(self.original_cols - 1, -1, -1):
            if all(matrix[i][j] == -1 for i in range(self.original_rows)):
                end_col = j - 1
            else:
                break
        return end_col

    def fold(self, direction, line=None, diagonal_points=None):
        """
        Performs a single fold operation. 'direction' can be 'horizontal', 'vertical', or 'diagonal'.

        Args:
            direction (str): The direction of the fold ('horizontal', 'vertical', or 'diagonal').
            line (int, optional): The fold line for horizontal or vertical folds. Defaults to None.
            diagonal_points (tuple, optional): A tuple of two points defining the diagonal fold line. Defaults to None.
        
        Raises:
            ValueError: If the fold line is out of bounds or if a diagonal fold is attempted without
                        specifying `diagonal_points`.
        """
        if direction == 'horizontal':
            if not (0 < line < self.current_rows):
                raise ValueError(f"Horizontal fold line {line} is out of bounds for {self.current_rows} rows")
            self.folds.append(('horizontal', line, self.current_rows, self.current_cols))
            new_rows = max(line, self.current_rows - line)
            new_grid = np.zeros((new_rows, self.current_cols))

            if line != new_rows:  # Folding from top to bottom
                start_row = self.get_start_row()
                real_line = start_row + line
                for i in range(self.original_rows):
                    for j in range(self.original_cols):
                        if start_row <= i < real_line:
                            self.complete_grid[i][j] = -1
            else:  # Folding from bottom to top
                start_row = self.get_start_row()
                end_row = self.get_end_row()
                real_line = start_row + line
                for i in range(self.original_rows):
                    for j in range(self.original_cols):
                        if real_line <= i <= end_row:
                            self.complete_grid[i][j] = -1

            self.grid = new_grid
            self.current_rows = new_rows

        elif direction == 'vertical':
            if not (0 < line < self.current_cols):
                raise ValueError(f"Vertical fold line {line} is out of bounds for {self.current_cols} cols")
            self.folds.append(('vertical', line, self.current_rows, self.current_cols))
            new_cols = max(line, self.current_cols - line)
            new_grid = np.zeros((self.current_rows, new_cols))

            if line < self.current_cols - line:  # Folding from left to right
                start_col = self.get_start_col()
                real_line = start_col + line
                for i in range(self.original_rows):
                    for j in range(self.original_cols):
                        if start_col <= j < real_line:
                            self.complete_grid[i][j] = -1
            else:  # Folding from right to left
                start_col = self.get_start_col()
                end_col = self.get_end_col()
                real_line = start_col + line
                for i in range(self.original_rows):
                    for j in range(self.original_cols):
                        if real_line <= j <= end_col:
                            self.complete_grid[i][j] = -1

            self.grid = new_grid
            self.current_cols = new_cols

        elif direction == 'diagonal':
            if diagonal_points is None:
                raise ValueError("Diagonal fold requires two points to define the 45-degree line")
            (x1, y1), (x2, y2) = diagonal_points

            self.folds.append(('diagonal', diagonal_points, self.current_rows, self.current_cols))
            new_grid = np.zeros((self.current_rows, self.current_cols))
            
            start_row = self.get_start_row()
            start_col = self.get_start_col()
            end_row = self.get_end_row()
            end_col = self.get_end_col()
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            x1 += start_row
            x2 += start_row
            y1 += start_col
            y2 += start_col
            
            for i in range(self.current_rows):
                for j in range(self.current_cols):
                    if slope < 0:
                        if min(x1, x2) == start_row and min(y1, y2) == start_col:
                            if (slope * i + intercept) > j:
                                new_grid[i][j] = -1
                                self.complete_grid[start_row + i][start_col + j] = -1
                        elif max(x1, x2) == end_row and max(y1, y2) == end_col:
                            if (slope * i + intercept) < j:
                                new_grid[i][j] = -1
                                self.complete_grid[start_row + i][start_col + j] = -1
                    else:
                        if min(x1, x2) == start_row and max(y1, y2) == end_col:
                            if (slope * i + intercept) < j:
                                new_grid[i][j] = -1
                                self.complete_grid[start_row + i][start_col + j] = -1
                        elif max(x1, x2) == end_row and min(y1, y2) == start_col:
                            if (slope * i + intercept) > j:
                                new_grid[i][j] = -1
                                self.complete_grid[start_row + i][start_col + j] = -1
                                 
            self.grid = new_grid
        else:
            raise ValueError("Direction must be 'horizontal', 'vertical', or 'diagonal'")

    def punch(self, points):
        """
        Punches holes in the folded grid. 'points' are hole coordinates.

        Args:
            points (list): A list of (row, col) tuples representing the coordinates
                           of the holes to punch.

        Raises:
            ValueError: If a punch position is out of bounds or a hole already exists.
        """
        for point in points:
            x, y = point
            if not (0 <= x < self.current_rows and 0 <= y < self.current_cols and self.grid[x][y] == 0):
                raise ValueError(f"Punch position ({x}, {y}) is out of bounds for grid {self.current_rows}x{self.current_cols}")
            
            self.grid[x][y] = 1
            
            start_row = self.get_start_row()
            start_col = self.get_start_col()
            self.complete_grid[x + start_row][y + start_col] = 1
        
        self.folds.append(("punch", points, self.current_rows, self.current_cols))
            
    def unfold(self):
        """
        Unfolds the paper in reverse order and calculates the final hole pattern.
        """
        for fold in reversed(self.folds):
            direction, line, orig_rows, orig_cols = fold
            if direction == 'punch':
                continue
            if direction == 'diagonal':
                (x1, y1), (x2, y2) = line
            new_grid = np.zeros((orig_rows, orig_cols))    
            new_grid_visited = [[0 for _ in range(orig_cols)] for _ in range(orig_rows)]
            if direction == 'horizontal':
                if line == self.current_rows:
                    for i in range(line):
                        for j in range(orig_cols):
                            new_grid[i][j] = self.grid[i][j]
                            sym_i = 2 * line - 1 - i
                            if sym_i < orig_rows:
                                new_grid[sym_i][j] = self.grid[i][j]
                else:
                    for i in range(line, orig_rows):
                        for j in range(orig_cols):
                            new_grid[i][j] = self.grid[i - line][j]
                            sym_i = 2 * line - 1 - i
                            if sym_i >= 0:
                                new_grid[sym_i][j] = self.grid[i - line][j]
            elif direction == 'vertical':
                if line == self.current_cols:
                    for i in range(orig_rows):
                        for j in range(line):
                            new_grid[i][j] = self.grid[i][j]
                            sym_j = 2 * line - 1 - j
                            if sym_j < orig_cols:
                                new_grid[i][sym_j] = self.grid[i][j]
                else:
                    for i in range(orig_rows):
                        for j in range(line, orig_cols):
                            new_grid[i][j] = self.grid[i][j - line]
                            sym_j = 2 * line - 1 - j
                            if sym_j >= 0:
                                new_grid[i][sym_j] = self.grid[i][j - line]
            elif direction == 'diagonal':
                x_min, y_min = min(x1, x2), min(y1, y2)
                x_max, y_max = max(x1, x2), max(y1, y2)
                
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                for i in range(x_min, x_max + 1):
                    for j in range(y_min, y_max + 1):
                        if self.grid[i][j] == -1:
                            if slope > 0:
                                sim_i = int(j - intercept)
                                sim_j = int(i + intercept)
                                new_grid[i][j] = self.grid[sim_i][sim_j]
                            else:
                                sim_i = int(-j + intercept)
                                sim_j = int(-i + intercept)
                                new_grid[i][j] = self.grid[sim_i][sim_j]
                            new_grid_visited[i][j] = 1
                    
                for i in range(orig_rows):
                    for j in range(orig_cols):
                        if new_grid_visited[i][j] == 0: 
                            new_grid[i][j] = self.grid[i][j]
            
            self.grid = new_grid
            self.current_rows = orig_rows
            self.current_cols = orig_cols
        
        self.folds = []  # Clear fold record

    def print_grid(self):
        """
        Prints the current grid state, with '●' for holes and '○' for no holes.
        """
        for row in self.grid:
            row_line = []
            for cell in row:
                if cell == 1:
                    row_line.append('●')
                elif cell == 0:
                    row_line.append('○')
                elif cell == -1:
                    row_line.append(' ')
            print(' '.join(row_line))
        print()


def draw_paper(paper, matrixs, save_dir):
    """
    Draws the paper's folded state with boundaries and hole locations.

    Args:
        paper (Paper): The Paper object to draw.
        matrixs (list): A list of grid states at each step.
        save_dir (str): The directory to save the output image.
    """
    fig, ax = plt.subplots()

    def find_region_boundary(matrix, visited, find=-1):
        """
        Finds the bounding box of a region in the matrix.
        
        Args:
            matrix (np.ndarray): The grid matrix.
            visited (np.ndarray): A boolean matrix to track visited cells.
            find (int, optional): The value to search for. Defaults to -1 (folded region).
        
        Returns:
            tuple: A tuple containing the condition mask and the bounding box coordinates.
        """
        rows, cols = matrix.shape
        if find == -1:
            condition = (matrix == -1) & (~visited)
        else:
            condition = (matrix != -1) & (~visited)
        min_r, max_r, min_c, max_c = rows, 0, cols, 0
        for i in range(rows):
            for j in range(cols):
                if condition[i, j]:
                    visited[i][j] = True
                    min_r = min(min_r, i)
                    max_r = max(max_r, i)
                    min_c = min(min_c, j)
                    max_c = max(max_c, j)
        min_x, max_x, min_y, max_y = min_c, max_c + 1, min_r, max_r + 1
        return condition, [min_x, max_x, min_y, max_y]

    def find_symmetric_line(corners, direction, line=None):
        """
        Calculates the symmetric boundary based on the fold line.

        Args:
            corners (list): A list of corner coordinates [min_x, max_x, min_y, max_y].
            direction (str): The fold direction ('horizontal' or 'vertical').
            line (int, optional): The fold line. Defaults to None.
        
        Returns:
            list: The new symmetric boundary coordinates.
        """
        min_x, max_x, min_y, max_y = corners
        if direction == "horizontal":
            if line > min_y:
                temp = 2 * line - min_y
                min_y = max_y
                max_y = temp
            else:
                temp = 2 * line - max_y
                max_y = min_y
                min_y = temp
        elif direction == "vertical":
            if line > min_x:
                temp = 2 * line - min_x
                min_x = max_x
                max_x = temp
            else:
                temp = 2 * line - max_x
                max_x = min_x
                min_x = temp
        return [min_x, max_x, min_y, max_y]
            
    def draw_box(corners, find=-1):
        """
        Draws a rectangular boundary on the plot.

        Args:
            corners (list): A list of corner coordinates [min_x, max_x, min_y, max_y].
            find (int, optional): The region type to draw. Defaults to -1.
        """
        min_x, max_x, min_y, max_y = corners
        orig_region = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
        orig_x, orig_y = zip(*orig_region + [orig_region[0]])
        if find == -1:
            ax.plot(orig_x, orig_y, 'b--', linewidth=1.5, label='Original Region', alpha=0.3)
        elif find == 0:
            ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region')

    def draw_triangle(diagonal_points, corners):
        """
        Draws a triangular region for diagonal folds.

        Args:
            diagonal_points (tuple): A tuple of two points defining the diagonal fold line.
            corners (list): The corners of the bounding box.
        """
        def top_right():
            orig_region = [(diag_min_x, diag_max_y), (diag_min_x, diag_min_y), (diag_max_x, diag_max_y)]
            orig_x, orig_y = zip(*orig_region + [orig_region[0]])
            ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region') 
            if flag:
                orig_region = [(diag_min_x, diag_min_y), (min_x, min_y), (min_x, max_y), (max_x, max_y), (diag_max_x, diag_max_y)]
                orig_x, orig_y = zip(*orig_region)
                ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region')
            orig_region = [(diag_max_x, diag_min_y), (diag_min_x, diag_min_y), (diag_max_x, diag_max_y)]
            orig_x, orig_y = zip(*orig_region + [orig_region[0]])
            ax.plot(orig_x, orig_y, 'b--', linewidth=1.5, label='Original Region', alpha=0.3)

        def bottom_right():
            orig_region = [(diag_min_x, diag_min_y), (diag_max_x, diag_min_y), (diag_min_x, diag_max_y)]
            orig_x, orig_y = zip(*orig_region + [orig_region[0]])
            ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region')
            if flag:
                orig_region = [(diag_max_x, diag_min_y), (max_x, min_y), (min_x, min_y), (min_x, max_y), (diag_min_x, diag_max_y)]
                orig_x, orig_y = zip(*orig_region)
                ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region')
            orig_region = [(diag_max_x, diag_max_y), (diag_max_x, diag_min_y), (diag_min_x, diag_max_y)]
            orig_x, orig_y = zip(*orig_region + [orig_region[0]])
            ax.plot(orig_x, orig_y, 'b--', linewidth=1.5, label='Original Region', alpha=0.3)

        def top_left():
            orig_region = [(diag_max_x, diag_max_y), (diag_max_x, diag_min_y), (diag_min_x, diag_max_y)]
            orig_x, orig_y = zip(*orig_region + [orig_region[0]])
            ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region')
            if flag:
                orig_region = [(diag_max_x, diag_min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (diag_min_x, diag_max_y)]
                orig_x, orig_y = zip(*orig_region)
                ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region')
            orig_region = [(diag_min_x, diag_min_y), (diag_max_x, diag_min_y), (diag_min_x, diag_max_y)]
            orig_x, orig_y = zip(*orig_region + [orig_region[0]])
            ax.plot(orig_x, orig_y, 'b--', linewidth=1.5, label='Original Region', alpha=0.3)

        def bottom_left():
            orig_region = [(diag_max_x, diag_min_y), (diag_max_x, diag_max_y), (diag_min_x, diag_min_y)]
            orig_x, orig_y = zip(*orig_region + [orig_region[0]])
            ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region')
            if flag:
                orig_region = [(diag_max_x, diag_max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y), (diag_min_x, diag_min_y)]
                orig_x, orig_y = zip(*orig_region)
                ax.plot(orig_x, orig_y, 'b-', linewidth=3, label='Original Region')
            orig_region = [(diag_min_x, diag_max_y), (diag_max_x, diag_max_y), (diag_min_x, diag_min_y)]
            orig_x, orig_y = zip(*orig_region + [orig_region[0]])
            ax.plot(orig_x, orig_y, 'b--', linewidth=1.5, label='Original Region', alpha=0.3)
        
        (r1, c1), (r2, c2) = diagonal_points
        r1 += paper.get_start_row()
        r2 += paper.get_start_row()
        c1 += paper.get_start_col()
        c2 += paper.get_start_col()
        
        diag_min_x = min(c1, c2)
        diag_max_x = max(c1, c2) + 1
        diag_min_y = min(r1, r2)
        diag_max_y = max(r1, r2) + 1
        
        min_x, max_x, min_y, max_y = corners
        flag = True
        
        slope = (c1 - c2) / (r1 - r2)
        if slope < 0:
            if min(r1, r2) == paper.get_start_row() and min(c1, c2) == paper.get_start_col():
                top_left()
            elif max(r1, r2) == paper.get_end_row() and max(c1, c2) == paper.get_end_col():
                bottom_right()
        else:
            if min(r1, r2) == paper.get_start_row() and max(c1, c2) == paper.get_end_col():
                top_right()
            elif max(r1, r2) == paper.get_end_row() and min(c1, c2) == paper.get_start_col():
                bottom_left()

    def draw_punch(points):
        """
        Draws a circle to represent a punch hole.

        Args:
            points (list): A list of (row, col) tuples for the holes.
        """
        for point in points:
            r, c = point
            x, y = c, r
            circle = plt.Circle((x + 0.5, y + 0.5), 0.2, color='r', fill=False, linewidth=2)
            ax.add_patch(circle)
            
    rows, cols = paper.original_rows, paper.original_cols
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, linestyle='-', alpha=0.3, zorder=0)

    for i, fold in enumerate(paper.folds):
        direction, line_or_points, _, _ = fold
        matrix = np.array(matrixs[i + 1])
        if direction in ["horizontal", "vertical"]:
            _, corners_1 = find_region_boundary(matrix, paper.visited)
            draw_box(corners_1)
            if i == len(paper.folds) - 1 or (i == len(paper.folds) - 2 and paper.folds[i+1][0] == "punch"):
                _, corners_0 = find_region_boundary(matrix, np.zeros_like(matrix, dtype=bool), find=0)
                draw_box(corners_0, find=0)
                if direction == "horizontal":
                    line = paper.get_start_row(matrix=np.array(matrixs[i])) + line_or_points
                elif direction == "vertical":
                    line = paper.get_start_col(matrix=np.array(matrixs[i])) + line_or_points
                corner_symm_line = find_symmetric_line(corners_1, direction, line=line)
                draw_box(corner_symm_line, find=0)
        elif fold[0] == "diagonal":
            _, corners_0 = find_region_boundary(matrix, np.zeros_like(matrix, dtype=bool), find=0)
            draw_triangle(fold[1], corners_0)
        else:
            points = [(i, j) for i in range(paper.original_rows) for j in range(paper.original_cols) if paper.complete_grid[i][j] == 1]
            draw_punch(points)
    
    paper.visited = np.zeros((rows, cols), dtype=bool)
    ax.set_aspect('equal')
    plt.savefig(f"{save_dir}/{len(paper.folds)}_{fold[0]}.png", bbox_inches='tight')
    plt.tight_layout()
    plt.close()

def draw_unfold(paper, save_name, save_dir):
    """
    Draws the final unfolded grid with hole locations.

    Args:
        paper (Paper): The Paper object to draw.
        save_name (str): The name for the saved image file.
        save_dir (str): The directory to save the output image.
    """
    fig, ax = plt.subplots()
    rows, cols = paper.original_rows, paper.original_cols
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.grid(True, linestyle='-', alpha=0.3, zorder=0)

    for i in range(rows):
        for j in range(cols):
            if paper.grid[i][j] == 1:
                circle = plt.Circle((j + 0.5, i + 0.5), 0.2, color="black", fill=True)
                ax.add_patch(circle)
    
    ax.set_aspect('equal')
    plt.savefig(f"{save_dir}/unfold_{save_name}.png", bbox_inches='tight')
    plt.tight_layout()
    plt.close()

def find_45_degree_fold_endpoints(rows, cols):
    """
    Finds all valid 45-degree fold line endpoints on the paper's boundary.

    Args:
        rows (int): The number of rows in the paper grid.
        cols (int): The number of columns in the paper grid.

    Returns:
        list: A list of tuples, where each tuple contains two (row, col) points
              defining a valid diagonal fold.
    """
    endpoints = []
    
    def is_boundary(x, y):
        return x == 0 or x == rows - 1 or y == 0 or y == cols - 1
    
    def is_45_degree_line(x1, y1, x2, y2):
        return abs(x2 - x1) == abs(y2 - y1) and (x1 != x2 or y1 != y2)

    def are_on_adjacent_edges(x1, y1, x2, y2):
        top = (x1 == 0 or x2 == 0)
        bottom = (x1 == rows - 1 or x2 == rows - 1)
        left = (y1 == 0 or y2 == 0)
        right = (y1 == cols - 1 or y2 == cols - 1)
        return (top and left) or (top and right) or (bottom and left) or (bottom and right)
    
    def is_valid_fold(x1, y1, x2, y2):
        if not (is_boundary(x1, y1) and is_boundary(x2, y2)):
            return False
        if not is_45_degree_line(x1, y1, x2, y2):
            return False
        if not are_on_adjacent_edges(x1, y1, x2, y2):
            return False
        return True
    
    for x1 in range(rows):
        for y1 in range(cols):
            for x2 in range(rows):
                for y2 in range(cols):
                    if (x1, y1) != (x2, y2) and is_valid_fold(x1, y1, x2, y2):
                        if ((x1, y1), (x2, y2)) not in endpoints and ((x2, y2), (x1, y1)) not in endpoints:
                            endpoints.append(((x1, y1), (x2, y2)))
    return endpoints

def random_generate_data(rows=5, cols=5, steps=5, punches=2, save_dir=None, case_id=0):
    """
    Generates a single dataset (images and JSON file) for a paper folding problem.

    Args:
        rows (int, optional): Number of rows for the paper grid. Defaults to 5.
        cols (int, optional): Number of columns for the paper grid. Defaults to 5.
        steps (int, optional): Number of folding steps. Defaults to 5.
        punches (int, optional): Number of holes to punch. Defaults to 2.
        save_dir (str, optional): The directory to save the dataset. Defaults to None.
        case_id (int, optional): The case ID for naming purposes. Defaults to 0.

    Raises:
        ValueError: If the number of zero positions is less than the number of punches.
    """
    matrixs = []
    paper = Paper(rows, cols)
    matrixs.append(copy.deepcopy(paper.complete_grid))
    
    for step in range(steps):
        if step != steps - 1:
            direction = random.choice(["horizontal", "vertical"])
        else:
            direction = random.choice(["diagonal"])
        
        if direction == "horizontal":
            line = random.randint(1, paper.current_rows - 1)
            paper.fold(direction, line=line)
        elif direction == "vertical":
            line = random.randint(1, paper.current_cols - 1)
            paper.fold(direction, line=line)
        else:
            diagonal_points = random.choice(find_45_degree_fold_endpoints(paper.current_rows, paper.current_cols))
            paper.fold(direction, diagonal_points=diagonal_points)
        paper.print_grid()
        matrixs.append(copy.deepcopy(paper.complete_grid))
        draw_paper(paper=paper, matrixs=matrixs, save_dir=save_dir)
    
    zero_positions = [(i, j) for i in range(paper.current_rows) for j in range(paper.current_cols) if paper.grid[i][j] == 0]
    if len(zero_positions) < punches:
        raise ValueError("Number of zero positions is less than punches, cannot select enough points.")
    points = random.sample(zero_positions, punches)
    
    paper.punch(points)
    paper.print_grid()
    matrixs.append(copy.deepcopy(paper.complete_grid))
    draw_paper(paper=paper, matrixs=matrixs, save_dir=save_dir)
    json_data = {"folds": paper.folds}
    
    choices = ["A", "B", "C", "D"]
    random.shuffle(choices)
    explanations = {}
    
    paper.unfold()
    paper.print_grid()
    draw_unfold(paper=paper, save_name=f"correct_{choices[0]}", save_dir=save_dir)
    
    row_or_col = random.choice(['row', 'col'])
    rows_with_ones = [i for i in range(paper.current_rows) if 1 in paper.grid[i]]
    cols_with_ones = [j for j in range(paper.current_cols) if 1 in paper.grid[:, j]]
    
    if len(rows_with_ones) == paper.current_rows and len(cols_with_ones) == paper.current_cols:
        row_or_col = 'both'
    elif not rows_with_ones:
        row_or_col = 'col'
    elif not cols_with_ones:
        row_or_col = 'row'
        
    if row_or_col == 'row':
        row_to_zero = random.choice(rows_with_ones)
        new_paper = Paper(rows, cols)
        new_paper.grid = copy.deepcopy(paper.grid)
        new_paper.grid[row_to_zero] = [0 for _ in range(cols)]
        draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[1]}", save_dir=save_dir)
        explanations[choices[1]] = f"Option {choices[1]} is incorrect because it is missing holes on row {row_to_zero + 1}."
        
        rows_with_all_zeros = [i for i in range(paper.current_rows) if 1 not in paper.grid[i]]
        if rows_with_all_zeros:
            row_to_one = random.choice(rows_with_all_zeros)
            new_paper = Paper(rows, cols)
            new_paper.grid = copy.deepcopy(paper.grid)
            new_paper.grid[row_to_one] = paper.grid[row_to_zero]
            draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[2]}", save_dir=save_dir)
            explanations[choices[2]] = f"Option {choices[2]} is incorrect because it adds extra holes on row {row_to_one + 1}."
            
            new_paper = Paper(rows, cols)
            new_paper.grid = copy.deepcopy(paper.grid)
            new_paper.grid[row_to_one] = paper.grid[row_to_zero]
            new_paper.grid[row_to_zero] = [0 for _ in range(cols)]
            draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[3]}", save_dir=save_dir)
            explanations[choices[3]] = f"Option {choices[3]} is incorrect because the holes that should be on row {row_to_zero + 1} are shown on row {row_to_one + 1}."
    
    elif row_or_col == 'col':
        col_to_zero = random.choice(cols_with_ones)
        new_paper = Paper(rows, cols)
        new_paper.grid = copy.deepcopy(paper.grid)
        new_paper.grid[:, col_to_zero] = 0
        draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[1]}", save_dir=save_dir)
        explanations[choices[1]] = f"Option {choices[1]} is incorrect because it is missing holes on column {col_to_zero + 1}."

        cols_with_all_zeros = [j for j in range(paper.current_cols) if 1 not in paper.grid[:, j]]
        if cols_with_all_zeros:
            col_to_one = random.choice(cols_with_all_zeros)
            new_paper = Paper(rows, cols)
            new_paper.grid = copy.deepcopy(paper.grid)
            new_paper.grid[:, col_to_one] = paper.grid[:, col_to_zero]
            draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[2]}", save_dir=save_dir)
            explanations[choices[2]] = f"Option {choices[2]} is incorrect because it adds extra holes on column {col_to_one + 1}."

            new_paper = Paper(rows, cols)
            new_paper.grid = copy.deepcopy(paper.grid)
            new_paper.grid[:, col_to_one] = paper.grid[:, col_to_zero]
            new_paper.grid[:, col_to_zero] = 0
            draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[3]}", save_dir=save_dir)
            explanations[choices[3]] = f"Option {choices[3]} is incorrect because the holes that should be on column {col_to_zero + 1} are shown on column {col_to_one + 1}."
    
    elif row_or_col == 'both':
        row_to_zero = random.choice(rows_with_ones)
        new_paper = Paper(rows, cols)
        new_paper.grid = copy.deepcopy(paper.grid)
        new_paper.grid[row_to_zero] = [0 for _ in range(cols)]
        draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[1]}", save_dir=save_dir)
        explanations[choices[1]] = f"Option {choices[1]} is incorrect because it is missing holes on row {row_to_zero + 1}."
        
        if len(cols_with_ones) > 1:
            col_to_zeros = random.sample(cols_with_ones, 2)
            new_paper = Paper(rows, cols)
            new_paper.grid = copy.deepcopy(paper.grid)
            new_paper.grid[:, col_to_zeros[0]] = 0
            draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[2]}", save_dir=save_dir)
            explanations[choices[2]] = f"Option {choices[2]} is incorrect because it is missing holes on column {col_to_zeros[0] + 1}."
            
            new_paper = Paper(rows, cols)
            new_paper.grid = copy.deepcopy(paper.grid)
            new_paper.grid[:, col_to_zeros[1]], new_paper.grid[:, col_to_zeros[0]] = new_paper.grid[:, col_to_zeros[0]].copy(), new_paper.grid[:, col_to_zeros[1]].copy()
            draw_unfold(paper=new_paper, save_name=f"incorrect_{choices[3]}", save_dir=save_dir)
            explanations[choices[3]] = f"Option {choices[3]} is incorrect because the holes in columns {col_to_zeros[1] + 1} and {col_to_zeros[0] + 1} are swapped."
    
    json_data['explanation'] = explanations
    with open(f"{save_dir}/info.json", "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)
    
def generate_datasets(args):
    """
    Main function to generate N datasets.

    Args:
        args (argparse.Namespace): An object containing command-line arguments,
                                   including N, rows, cols, steps, punches, and output_dir.
    """
    for i in range(args.N):
        save_dir = os.path.join(args.output_dir, f"dataset_{i+1}")
        os.makedirs(save_dir, exist_ok=True)
        try:
            random_generate_data(rows=args.rows, cols=args.cols, steps=args.steps, punches=args.punches, save_dir=save_dir, case_id=i)
            print(f"Dataset {i+1} generated successfully in {save_dir}")
        except ValueError as e:
            print(f"Skipping dataset {i+1} due to an error: {e}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate N paper folding datasets.")
    parser.add_argument("--N", type=int, default=1, help="Number of datasets to generate.")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows for the paper grid.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns for the paper grid.")
    parser.add_argument("--steps", type=int, default=3, help="Number of folding steps.")
    parser.add_argument("--punches", type=int, default=3, help="Number of holes to punch.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to save the datasets, e.g., /path/to/output_folder")
    
    args = parser.parse_args()
    generate_datasets(args)
    
"""
    python your_script_name.py 
        --N 10 
        --rows 5 
        --cols 5 
        --steps 3 
        --punches 3 -
        -output_dir /path/to/your/output/folder
"""