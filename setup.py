import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frechet-audio-distance", 
    version="0.0.1",
    author="Hao Hao Tan",
    description="A lightweight library of Frechet Audio Distance calculation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=["frechet-audio-distance"],
    package_dir={'':'src'},
    install_requires=[]
)