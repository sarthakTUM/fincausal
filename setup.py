from distutils.core import setup

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='fnp',
    version='0.0.1',
    license='',
    long_description=open('README.txt').read(),
    packages=find_packages(exclude="tests"),  # same as name
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.6",
)