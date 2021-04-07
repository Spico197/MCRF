import setuptools

from mcrf import __version__


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='pytorch-mcrf',
    version=__version__,
    author="Tong Zhu",
    author_email="tzhu1997@outlook.com",
    description="Multiple CRF implementation for PyTorch",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages('.', exclude=('*tests*',)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
    install_requires=[
    ],
    package_data={
    },
    include_package_data=False,
)
