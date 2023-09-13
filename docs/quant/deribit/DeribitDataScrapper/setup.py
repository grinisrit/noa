from setuptools import setup, find_packages

import json
import os

module_name = "scrapperDeribit"


file_dir = os.path.dirname(os.path.realpath(__file__))
absdir = lambda p: os.path.join(file_dir, p)


# open readme and convert all relative path to absolute path
with open("README.md", "r") as f:
    long_desc = f.read()

def get_requirements(fname):
    with open(absdir(fname), "r") as f:
        return [line.strip() for line in f.read().split("\n") if line.strip() != ""]



if __name__ == '__main__':
    print(find_packages())
    setup(
        name=module_name,
        version='0.0.0029',
        description='Deribit Scrapping Interface',
        long_description=long_desc,
        long_description_content_type="text/markdown",
        author='Molozey',
        author_email='molozeyWorking@gmail.com',
        packages=find_packages('src'),
        package_dir={'': 'src'},
        # include_package_data=True,
        python_requires=">=3.10",
        install_requires=get_requirements("req.txt"),
        classifiers=[
            "Programming Language :: Python :: 3.10",
        ],
        keywords="data-scrapper deribit crypto",
        zip_safe=False,
        include=["test/*", "examples/*"]
    )