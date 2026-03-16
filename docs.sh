#!/usr/bin/env bash
# Generate API documentation (JSON + Markdown + llms-full.txt) from Python docstrings
set -e
cd "$(dirname "$0")/docs"
python parse.py -i ../src/ptspy -o ./src/assets/docs.json -m ./public/docs.md \
  -p ./llms-preamble.md -l ./public/llms-full.txt
echo "Generated docs/src/assets/docs.json, docs/public/docs.md, and docs/public/llms-full.txt"
