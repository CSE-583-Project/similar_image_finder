## Similiar Image Finder<br/>

A similar image finder tool for customers who wish to find products similar to one which they have an image of.

## How to use the GUI

## Installation
1. Open a terminal.
2. Clone the repoistory using `git clone https://github.com/CSE-583-Project/similar_image_finder.git`.
3. Change to the similar_image_finder directory using `cd similar_image_finder`.
4. Set up a new virtual environment with all necessary packages and their dependencies using `conda env create -f environment.yml`.
5. Activate the virtual environment with `conda activate env`.
6. Deactivate the virtual environment using `conda deactivate`.

## Repository Structure
 ```
.
├── doc
├── similar_image_finder
│   ├── data_loader
│   ├── gui_module
│   ├── model
│   └── inference.py
│   ├── train.py
├── tests
│   ├── test_loader.py
├── LICENSE
├── README.md
└── environment.yml
 ```

## Tests

[![Python Package using Conda](https://github.com/CSE-583-Project/similar_image_finder/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/CSE-583-Project/similar_image_finder/actions/workflows/python-package-conda.yml)