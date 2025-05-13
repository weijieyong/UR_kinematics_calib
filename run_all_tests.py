#!/usr/bin/env python3
import csv
import subprocess
import sys
import os

def main():
    # locate paths relative to this script
    root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(root, 'data', 'joint-eef-data.csv')
    script_path = os.path.join(root, 'dry_parse.py')

    # open CSV and iterate rows
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Uncomment the next line to skip header row
        # next(reader, None)
        for row in reader:
            line = ','.join(row)
            print(f"Running test for: {line}")
            # invoke dry_parse.py for each line
            proc = subprocess.run(
                [sys.executable, script_path, "--compare", line],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # print outputs
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)

    print("All tests completed.")

if __name__ == '__main__':
    main()
