from setuptools import find_packages, setup

setup(
    name="Generative AI Project",
    version='0.0.0',
    author="Abhishek Chaudhari",
    author_email="abhishekchaudhari336@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
