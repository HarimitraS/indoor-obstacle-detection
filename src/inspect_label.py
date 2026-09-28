import glob

label_file = glob.glob(
    "dataset/raw/labels/*.txt"
)[0]

with open(label_file, "r") as f:
    values = [int(x) for x in f.read().split()]

print("=" * 60)
print("YOLIC LABEL INSPECTION")
print("=" * 60)

print(f"\nFile: {label_file}")
print(f"Total values: {len(values)}")
print(f"Expected: 210")

print("\nValues grouped by CoI:")
print("-" * 60)

for i in range(30):

    start = i * 7
    end = start + 7

    cell = values[start:end]

    print(
        f"CoI {i+1:02d}: {cell}"
    )