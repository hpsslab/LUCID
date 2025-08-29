import sys


# Read all input 
input_text = ""
with open(sys.argv[1]) as file:
    input_text += file.read()

# Split into sections (tables) separated by empty lines
sections = input_text.split('\n\n')

all_grids = ""
for section in sections:
    lines = [line.strip() for line in section.split('\n') if line.strip()]
    if not lines:
        continue
    
    # Extract table name
    if lines[0].startswith('Table: '):
        table_name = lines[0][len('Table: '):].strip()
        entries_lines = lines[1:]
    else:
        table_name = lines[0].strip()
        entries_lines = lines[1:]
    
    # Process entries and collect all papers
    entries = []
    all_papers = set()
    for line in entries_lines:
        parts = line.split(',')
        # Extract X and Y from "Paper X" and "Paper Y"
        x = int(parts[0].split()[-1])
        y = int(parts[1].split()[-1])
        value = float(parts[2])
        entries.append((x, y, value))
        all_papers.add(x)
        all_papers.add(y)
    
    # Generate row and column orders
    # Row order: sorted ascending, 0 comes first if present
    row_order = sorted(all_papers)
    # Column order: non-zero papers sorted descending, then 0 at the end if present
    non_zero = sorted((p for p in all_papers if p != 0), reverse=True)
    if 0 in all_papers:
        column_order = non_zero + [0]
    else:
        column_order = non_zero
    
    # Create dictionaries to map papers to their indices
    row_index = {paper: idx for idx, paper in enumerate(row_order)}
    col_index = {paper: idx for idx, paper in enumerate(column_order)}
    
    # Initialize grid with 0.0
    n = len(row_order)
    grid = [[0.0 for _ in column_order] for _ in row_order]
    
    # Populate the grid with entries
    for x, y, val in entries:
        grid[row_index[x]][col_index[y]] = val
    
    # Set diagonal cells (same paper in row and column) to 1.00
    for paper in all_papers:
        grid[row_index[paper]][col_index[paper]] = 1.0
    
    # Zero out cells below the anti-diagonal (i + j > n - 1)
    for i in range(n):
        for j in range(n):
            if i + j > n - 1:
                grid[i][j] = None
    
    # Generate the output for the current grid
    print(f"Grid: {table_name.lower()},")
    all_grids += f"Grid: {table_name.lower()}, \n"
    # Create the header line
    headers = [f"{p}_response.txt" for p in column_order]
    print("\t, " + ", ".join(headers) + ",")
    all_grids += "\t, " + ", ".join(headers) + ",\n"
    # Create each row line
    for idx, paper in enumerate(row_order):
        row_name = f"{paper}_response.txt"
        # Format each value to two decimal places
        values = [f"{grid[idx][col_idx]}" for col_idx in range(n)]
        print(f"\t{row_name}, " + ", ".join(values) + ",")
        all_grids += f"\t{row_name}, " + ", ".join(values) + ",\n"

    # Print a blank line after each grid for separation
    print()
    all_grids  += "\n\n"

with open(sys.argv[2], "w") as file:
    file.write(all_grids)

