## Similiar Image Finder<br/>

A similar image finder tool for customers who wish to find products similar to one of which they have an image of.

## User Interface
The GUI has two pages for the Merchant/User:

### Item Upload Page
This page allows a Merchant to add an item to the database. In order to do this the Merchant must upload an image in .jpg/.png/.jpeg format and define the category to the item that should be added. Currently, there are only four categories based on item type and gender.

### Similar Item Finder
A user can get similar image recommendations on this page. They simply have to upload an image and our custom-trained Machine Learning model will compute the similarity between the input image and the items already present in the database. A processing time of about 30 secs is expected to run the ML model on the input image.

## Example of using the GUI
[GUI Demo](https://www.youtube.com/watch?v=XPiMd6PHYoc)

## Using the GUI
1. Install [Streamlit](https://docs.streamlit.io/library/get-started/installation) to run the GUI
2. Install [PyDrive](https://pypi.org/project/PyDrive/) to access the database on GDrive
3. Run the following command on the terminal from the similar_image_finder directory: `streamlit run gui_module/merchant_upload_page.py`

## Access to the Database
In order to upload images or get similar images the user needs access to the database hosted on the Google Cloud Platform. If you are testing the GUI or want to use the service get yourself added to the list of verified users. Contact the developer to get added ([email](kb111198@uw.edu))

## Installation
1. Open a terminal.
2. Clone the repoistory using `git clone https://github.com/CSE-583-Project/similar_image_finder.git`.
3. Change to the similar_image_finder directory using `cd similar_image_finder`.
4. Set up a new virtual environment with all necessary packages and their dependencies using `conda env create -f environment.yml`.
5. Activate the virtual environment with `conda activate cse583`.
6. Deactivate the virtual environment using `conda deactivate`.

## Repository Structure
 ```
.
├── doc
├── similar_image_finder
│   ├── data_loader
│   ├── gui_module
│   ├── model
│   ├── tests
│   ├── train.py
│   └── inference.py
├── LICENSE
├── README.md
└── environment.yml
 ```

## Tests

[![Python Package using Conda](https://github.com/CSE-583-Project/similar_image_finder/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/CSE-583-Project/similar_image_finder/actions/workflows/python-package-conda.yml)
