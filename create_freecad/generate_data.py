from mental_rotation_tasks_utils import *
from metal_folding_tasks_utils import *
from visual_penetration_tasks_utils import *
from mental_animation_tasks_utils import *

save_folder = "path/to/save"
dirs = [dir_name for dir_name in os.listdir(save_folder)]
dirs = sorted(dirs, key=lambda x: int(x.split('-')[0]))
for dir_name in dirs:
    dir_path = f"{save_folder}/{dir_name}"
    
    # --- Mental Rotation ---
    # --- Example for 3ViewProj-CAD ---
    # prepare_composite_image_CAD(dir_path)

    # --- Example for 3ViewProj-Cubes ---
    # prepare_composite_image_Cubes3View(dir_path)
    
    # --- Example for 2D Rotation ---
    # prepare_composite_image_2DRotation(dir_path)

    # --- Example for 3D Rotation ---
    # prepare_composite_image_3DRotation(dir_path, mode=0) # 'Cannot be obtained' question
    # prepare_composite_image_3DRotation(dir_path, mode=1) # 'Can be obtained' question
    
    
    # --- Mental Folding ---
    # --- Example for Reconstructing a Cube from a Net (Simple Colors) ---
    # prepare_composite_image_CubeReconstruction(dir_path)

    # --- Example for Reconstructing a Cube from a Net (Complex Patterns) ---
    # prepare_composite_image_CubeReconstructionComplex(dir_path)

    # --- Example for Unfolding a Cube (Simple Colors) ---
    # prepare_composite_image_CubeUnfolding(dir_path)

    # --- Example for Unfolding a Cube (Complex Patterns) ---
    # prepare_composite_image_CubeUnfoldingComplex(dir_path)
    
    # --- Example for Paper Folding ---
    # prepare_composite_image_PaperFolding(dir_path)
    
    
    # --- Visual Penetration ---
    # --- Example for Cube Assembly ---
    # prepare_composite_image_CubeAssembly(dir_path)

    # --- Example for Cross-Section ---
    # prepare_composite_image_CrossSection(dir_path)
    
    # --- Example for Cube Counting (2-view or 3-view) ---
    # prepare_composite_image_CubeCounting(dir_path, mode=0) # 2-view
    # prepare_composite_image_CubeCounting(dir_path, mode=1) # 3-view
    
    
    # --- Mental Animation ---
    # --- Example for Arrow Moving (Simple) ---
    # prepare_composite_image_ArrowMoving(dir_path)

    # --- Example for Arrow Moving (Complex) ---
    # prepare_composite_image_ArrowMoving_complex(dir_path, mode=0) # Path question
    # prepare_composite_image_ArrowMoving_complex(dir_path, mode=1) # Final state question
    
    # --- Example for Block Moving Puzzle ---
    # prepare_composite_image_BlocksMoving(dir_path)
