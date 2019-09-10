import os
import argparse

# https://www.datendieter.de/item/Liste_von_deutschen_Staedtenamen_.csv

parser = argparse.ArgumentParser(description="Strips text out of quotes at each line from the input file")

parser.add_argument("filename", action="store", type=str)

args = parser.parse_args()

with open(args.filename, "r") as f:
    lines = f.readlines()

with open(args.filename, "w") as f:
    for line in lines:
        if "\"" in line:
            f.write(line.replace("\"", ""))
        else:
            f.write(line)