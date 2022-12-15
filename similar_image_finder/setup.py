from setuptools import setup
setup(
    name='similar-image-finder',
    version='1.0',
    author='Kush Bhatia, Sadjyot Gangolli, Pranav Kamath, Ayush Singh',
    description='A similar image finder tool for customers who wish to find products similar to one of which they have an image of.',
    long_description='',
    url='https://github.com/CSE-583-Project/similar_image_finder.git',
    keywords='development, setup, setuptools',
    python_requires='>=3.7, <4',
    packages=['distutils', 'distutils.command'],
    install_requires=[
        'PyYAML',
        'numpy',
        'numpy-base',
        'pandas',
        'pillow',
        'pip',
        'pylint',
        'python',
        'pytorch',
        'readline',
        'requests',
        'scikit-learn',
        'scipy',
        'setuptools',
        'streamlit',
        'torchvision',
        'tqdm',
        'urllib3',
        'wheel'
    ],
    package_data={
        'sample': ['similar_image_finder/tests/test_data/fashion_test.csv'],
    },
    entry_points={
        'runners': [
            'sample=sample:main',
        ]
    }
)