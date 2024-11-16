#!/bin/bash

SEARCH_ROOT="../apps/OursTracesCollection"

CMD="./process_sass_dir --dir"

find "$SEARCH_ROOT" -type d -name "sass_traces" | while read dir; do
    echo "Processing directory: $dir"
    $CMD "$dir"
    find $dir -regextype posix-extended -regex ".*/kernel_([0-9]+)_gwarp_id_([0-9]+)\.split\.sass" -exec realpath {} \; > $dir/content.txt
    echo "Current Time: $(date)"
done
