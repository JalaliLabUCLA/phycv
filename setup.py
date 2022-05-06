'''
This is for building the PhyCV library.
Without building the library, you have to have the code in a specific directory to run all the algorithms
After building it, PhyCV will appear as a Python package in your Python environment.
If you are installing PhyCV from pip, you don't need setup.py

'''

from setuptools import setup, find_packages

def readme():
    with open('README.md', "r") as f:
        README = f.read()
    return README

def install_requires():
    with open("requirements.txt", "r") as f:
        install_requires = [x.strip() for x in f.readlines()]
    return install_requires


VERSION = '0.0.3'

# Setting up
setup(
    name="phycv",
    version=VERSION,
    author="Jalali-Lab",
    author_email="ucla.photonics.lab@gmail.com",
    description="physics-inspired computer vision algorithms",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires(),
    keywords=['python', 'image processing', 'computational imaging','computer vision', 'physics-inspired algorithm'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
