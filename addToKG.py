from pathlib import Path

def addToKG(path: Path) -> None:
    # Add article to KG, base case
    if path.is_file():
        if path.suffix == ".pdf":
            print(f"This file {path.name} is a PDF.")
        else:
            print(f"This file {path} has an unrecognized extension. Maybe we should add functionality for this file type later.")

    # Recursively get to the files
    elif path.is_dir():
        for root, _, files in path.walk():
            # NOTE: loop through files only. A loop through directories results in double counting due to how walk works (learned that the hard way)
            for file in files:
                addToKG(root.joinpath(file))
    else:
        print(f"SNO. Skipping {path.name} and moving on.") 
