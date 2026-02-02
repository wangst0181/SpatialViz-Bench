import os
import json
import time
import argparse
import Part
import FreeCADGui
import TechDrawGui
from FreeCAD import Units


def filter_complex_obj(file1):
    """
    Reads a STEP file and extracts geometric properties like faces, edges, volume, and solids count.

    Args:
        file1 (str): The path to the STEP file.

    Returns:
        tuple: A tuple containing the number of faces, edges, volume, and solids.
    """
    doc = App.newDocument()
    shape = Part.Shape()
    shape.read(file1)
    part_obj = doc.addObject("Part::Feature", "ImportedPart")
    part_obj.Shape = shape
    faces = len(part_obj.Shape.Faces)
    volume = part_obj.Shape.Volume
    edges = len(part_obj.Shape.Edges)
    solids_count = len(part_obj.Shape.Solids)
    return faces, edges, volume, solids_count


def generate_info(step_files, info_path):
    """
    Generates a JSON file with geometric information for a list of STEP files.

    Args:
        step_files (list): A list of paths to the STEP files.
        info_path (str): The path where the generated JSON file will be saved.

    Returns:
        dict: A dictionary containing the geometric data for each file.
    """
    data = {}
    for i, file in enumerate(step_files):
        print(f"[INFO] Processing {i+1}/{len(step_files)}: {file}")
        try:
            faces, edges, volume, solids_count = filter_complex_obj(file)
            data[file] = {
                "faces": faces,
                "edges": edges,
                "volume": volume,
                "solids": solids_count,
            }
        except Exception as e:
            print(f"[ERROR] Failed on {file}: {e}")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"[INFO] Saved info to {info_path}")
    return data


def export_drawings(step_files, info, save_root, template_path, filters):
    """
    Exports 3-view PDF and isometric PNG drawings for valid STEP models based on specified filters.

    Args:
        step_files (list): A list of paths to the STEP files.
        info (dict): A dictionary containing geometric information for the files.
        save_root (str): The root directory to save the output drawings.
        template_path (str): The path to the TechDraw SVG template.
        filters (dict): A dictionary of filters (min_faces, max_faces, min_volume) to apply.
    """
    for file in step_files:
        if file not in info:
            continue
        props = info[file]

        # filtering based on args
        if (
            props["faces"] < filters["min_faces"]
            or props["faces"] > filters["max_faces"]
            or props["volume"] < filters["min_volume"]
            or props["solids"] != 1
        ):
            continue

        save_name, _ = os.path.splitext(os.path.basename(file))
        save_dir = os.path.join(save_root, save_name)
        os.makedirs(save_dir, exist_ok=True)

        doc = App.newDocument()

        shape = Part.Shape()
        shape.read(file)
        part_obj = doc.addObject("Part::Feature", "ImportedPart")
        part_obj.Shape = shape
        bbx = part_obj.Shape.BoundBox
        part_X, part_Y, part_Z = bbx.XLength, bbx.YLength, bbx.ZLength

        doc.recompute()

        view = FreeCADGui.ActiveDocument.ActiveView
        view.viewIsometric()
        FreeCADGui.updateGui()

        page = doc.addObject("TechDraw::DrawPage", "Drawing")
        template = doc.addObject("TechDraw::DrawSVGTemplate", "Template")
        template.Template = template_path
        page.Template = template
        page_width, page_height = page.Template.Width, page.Template.Height

        proj_group = doc.addObject("TechDraw::DrawProjGroup", "ProjGroup")
        page.addView(proj_group)
        proj_group.Source = [part_obj]
        proj_group.ProjectionType = "First Angle"
        proj_group.ScaleType = "Custom"
        proj_group.Scale = min(
            float(page_width) / (float(part_X) + float(part_Y)),
            float(page_height) / (float(part_Z) + float(part_Y)),
        ) * 0.8

        proj_group.addProjection("Front")
        proj_group.addProjection("Left")
        proj_group.addProjection("Top")

        front_view_width = Units.Quantity(part_X * proj_group.Scale, "mm")
        front_view_height = Units.Quantity(part_Z * proj_group.Scale, "mm")

        proj_group.X = front_view_width / 2 + proj_group.spacingX * 1.0
        proj_group.Y = page_height - (front_view_height / 2 + proj_group.spacingY)

        doc.recompute()
        FreeCADGui.updateGui()
        time.sleep(0.5)

        pdf_path = os.path.join(save_dir, save_name + "_3View.pdf")
        TechDrawGui.exportPageAsPdf(page, pdf_path)

        view.fitAll()
        png_path = os.path.join(save_dir, save_name + "_Isometric.png")
        view.saveImage(png_path, 1280, 1024)

        print(f"[INFO] Exported: {pdf_path}, {png_path}")


def main(args):
    """
    Main function to orchestrate the process of generating info and exporting drawings.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    step_files = [
        os.path.join(args.cad_folder, f)
        for f in os.listdir(args.cad_folder)
        if f.endswith(".step")
    ]

    os.makedirs(args.info_folder, exist_ok=True)
    info_path = os.path.join(args.info_folder, f"{args.idx}-info.json")

    # Step 1: generate info
    if not os.path.exists(info_path) or args.force_regen:
        info = generate_info(step_files, info_path)
    else:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    # Step 2: export filtered drawings
    save_root = os.path.join(args.save_folder, args.idx)
    os.makedirs(save_root, exist_ok=True)

    filters = {
        "min_faces": args.min_faces,
        "max_faces": args.max_faces,
        "min_volume": args.min_volume,
    }
    export_drawings(step_files, info, save_root, args.template, filters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STEP to PDF/PNG exporter with info generation.")
    parser.add_argument("--idx", type=str, required=True, help="Model index (e.g. 0010).")
    parser.add_argument("--cad_folder", type=str, required=True, help="Path to CAD STEP folder.")
    parser.add_argument("--info_folder", type=str, required=True, help="Path to save JSON info files.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to save outputs.")
    parser.add_argument("--template", type=str, required=True, help="Path to TechDraw SVG template.")
    parser.add_argument("--force_regen", action="store_true", help="Force regenerate info JSON even if exists.")
    parser.add_argument("--min_faces", type=int, default=7, help="Minimum number of faces to keep.")
    parser.add_argument("--max_faces", type=int, default=20, help="Maximum number of faces to keep.")
    parser.add_argument("--min_volume", type=float, default=0.001, help="Minimum volume to keep.")
    args = parser.parse_args()
    main(args)

"""
    # run in FreeCAD
    import sys
    sys.argv = [
        "create_cad_3view_projection.py",
        "--idx", "0010",
        "--cad_folder", "/path/to/cad_model/0010",
        "--info_folder", "/path/to/info",
        "--save_folder", "/path/to/save",
        "--template", "/path/to/FreeCAD/data/Mod/TechDraw/Templates/A4_Landscape_blank.svg",
        "--min_faces", "7",
        "--max_faces", "20",
        "--min_volume", "0.001",
        "--force_regen"
    ]
    exec(open("/path/to/create_cad_3view_projection.py").read())

"""