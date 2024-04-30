import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="academy_reports",
    version="0.0.1",
    author="Balma Serrano",
    author_email="balmaserrano@gmail.com",
    description="Package to parse data and create reports",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://delaRochaLab@bitbucket.org/delaRochaLab/academy_reports.git",
    install_requires=[],
    include_package_data=True,
    packages=setuptools.find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['academy_reports=academy_reports.__main__:main']
    }
)
