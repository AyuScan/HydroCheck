import csv

input_filename = "datas.txt"  # Replace with your text file path
output_filename = "hydrocheck_dataset.csv"

parsed_rows = []
timestamp = 0

with open(input_filename, "r") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Parse key:value pairs from the line
        row_dict = dict(pair.split(":", 1) for pair in line.split())

        # Prepend the timestamp column
        full_row = {"Timestamp (s)": timestamp}
        full_row.update(row_dict)

        parsed_rows.append(full_row)

        # Increment timestamp by 6 seconds for the next line
        timestamp += 6

# Write to CSV
if parsed_rows:
    headers = parsed_rows[0].keys()

    with open(output_filename, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(parsed_rows)

    print(f"Successfully exported {len(parsed_rows)} rows to {output_filename}")