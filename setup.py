import setuptools

setuptools.setup(
    name="linsdex",
    version="0.1.0",
    author="Edmond Cunningham",
    author_email="edmondcunnin@cs.umass.edu",
    description="Inference in LTI-SDEs using CRFs",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    # url="https://github.com/EddieCunningham/generax",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=open("requirements.txt").read().splitlines(),
)