### Data Generation

This repository's data generation process is organized as follows:

* The directories within the `create_freecad` folder that are prefixed with `create_` contain the raw data generation code for each specific task.
* Other files in this directory are used to combine the raw data into the final input images.
* The main entry point for data generation is the `generate_data.py` script.
* **Important:** All tasks that require FreeCAD integration must have their code executed directly from the Python console within the FreeCAD application.