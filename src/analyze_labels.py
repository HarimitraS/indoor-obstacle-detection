import glob
from collections import Counter

label_files = glob.glob(
    "dataset/raw/labels/*.txt"
)

print("Total label files:", len(label_files))

print("\nAnalyzing first 100 labels...\n")

all_cells = []

for file in label_files[:100]:

    with open(file, "r") as f:
        values = [int(x) for x in f.read().split()]

    cells = [
        values[i:i+7]
        for i in range(0, 210, 7)
    ]

    all_cells.extend(cells)


print("Total cells analyzed:", len(all_cells))


print("\nPosition statistics")
print("=" * 60)

for position in range(7):

    values = [
        cell[position]
        for cell in all_cells
    ]

    count_ones = sum(values)

    count_zeros = len(values) - count_ones

    print(
        f"Position {position}: "
        f"1s = {count_ones}, "
        f"0s = {count_zeros}"
    )


print("\nExample cells")
print("=" * 60)

for i in range(30):

    print(
        f"Cell {i+1:02d}: "
        f"{all_cells[i]}"
    )