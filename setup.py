from setuptools import find_packages, setup

project_name = 'cardiovision-ai'
version = '0.1.0'
author = 'Muralidharan RMS'
description = "Computer vision for ECG-based myocardial infarction detection"


def get_requirements(file_path: str) -> list[str]:
    """
    Reads requirements.txt and returns a list of dependencies.
    """

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [req.replace('\n', '') for req in requirements]    

        if '-e .' in requirements:
            requirements.remove('-e .')

            return requirements


setup(
    name=project_name,
    version=version,
    author=author,
    description=description,
    packages=find_packages(),
    include_package_data=True,
    install_requires= get_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    license="MIT"    
)