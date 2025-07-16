#!/bin/bash

# Add all new files
git add .

# Commit with descriptive message
git commit -m "Add mapping results: DAG visualization and checkpoint data"

# Push to remote repository
git push

echo "Successfully committed and pushed mapping results!" 