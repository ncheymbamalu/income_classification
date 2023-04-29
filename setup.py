from setuptools import find_packages, setup

PYTHON_PACKAGES = [
    package
    for package in open(r"./requirements.txt", "r").read().splitlines()
    if package != "-e ."
]

setup(
    name="income_evaluation",
    version="0.0.1",
    author="Nchey Mbamalu",
    author_email="nchey.learnings@gmail.com",
    packages=find_packages(),
    install_requires=PYTHON_PACKAGES
)