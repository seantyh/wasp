import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wasp", # Replace with your own username
    version="0.1.0",
    author="Sean Tseng",
    author_email="seantyh@gmail.com",
    description="Word sense action pair",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seantyh/wasp",
    packages=['wasp'],
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)