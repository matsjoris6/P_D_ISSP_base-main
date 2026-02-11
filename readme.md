# P&D ISSP base scripts
This repository contains basic utilities for the P&D ISSP KU Leuven course.

## Setup:

1. Get a copy of the repository: either clone/fork this repo or just download a copy and extract in your own repository. 
2. Set up the environment:
   - Manually: ensure a moderately recent version of Python is installed (3.12+), create a virtual environment (`python -m venv .venv` in the appropriate folder, then activate it) and install the required dependencies (`pip install -r requirements.txt`). Note that one of the required packages, `pyroomacoustics`, requires having a C++ compiler and runtime, so ensure this is installed on your system.
   - Alternatively a `pixi.lock` file is required to automate setting up the environment; [install pixi](https://pixi.prefix.dev/latest/installation/) and run `pixi install` to get all required dependencies.
3. All set, you should be able to run start running code now.

## Folder structure
- `./package/gui_utils.py`: utils for the GUI. Not to be modified.
- `./package/general.py`: general functions, not GUI-related. You can add functions to that file to your convenience. Feel free to create other `.py` files as well.
- `./rirs/`: folder where your RIRs and acoustic scenario information will be stored (as Pickle archives: `name.pkl`).
- `./sound_files/`: folder containing provided sound files to conduct your tests.
- `./notebook_skeleton.ipynb`: skeleton notebook to start your work from.
- `./requirements.txt`: package requirements file.
- `./pixi.lock` and `./pixi.toml`: files to automate the installation of the environment.
