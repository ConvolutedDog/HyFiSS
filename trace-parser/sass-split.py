#!/bin/python
# -*- coding: utf-8 -*-

import os
import argparse

parser = argparse.ArgumentParser(description='Process sass dir.')

parser.add_argument('--dir', type=str, required=True,
                    help='The directory of sass files')

args = parser.parse_args()

sass_dir = args.dir
sass_dir = os.path.abspath(sass_dir)

files = os.listdir(sass_dir)
sass_files = [os.path.join(sass_dir, file) for file in files if (file.endswith(".sass") and not file.endswith(".split.sass"))]

f_open = {}
warp_content = {}

for sass_file in sass_files:
    print("Processing ", sass_file)
    content = open(sass_file, "r").read().split(" ")
    kernel_id = int(sass_file.split("/")[-1].split("_")[1].split(".sass")[0])

    for i in range(int(len(content)/3)):
        gwarp_id = int(content[i*3 + 2], 16)
        entry = (kernel_id, gwarp_id)
        
        # Use dictionaries to accumulate content instead of writing files directly
        if entry not in warp_content:
            warp_content[entry] = []
        warp_content[entry].append(content[i*3] + " " + content[i*3 + 1])

for (kernel_id, gwarp_id), lines in warp_content.items():
    file_path = os.path.join(sass_dir, "kernel_" + str(kernel_id) + "_gwarp_id_" + str(gwarp_id) + ".split.sass")
    with open(file_path, "w") as file:
        file.write("\n".join(lines))
