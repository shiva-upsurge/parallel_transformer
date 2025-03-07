from setuptools import setup, find_packages

setup(
    name="parallel-transformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    author="Shivanand",
    author_email="shivanand@upsurgelabs.com",
    description="A Parallel Transformer Implementation",
    python_requires=">=3.8",
)
