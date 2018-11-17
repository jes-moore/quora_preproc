import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quora_preproc",
    version="0.0.1",
    author="Jesse Moore",
    author_email="jessemoore07@gmail.com",
    description="Package for quora kaggle comp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jes-moore/quora_preproc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
