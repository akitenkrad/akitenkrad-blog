import sys

import setuptools

with open("README.md", mode="r") as f:
    long_description = f.read()

version_range_max = max(sys.version_info[1], 10) + 1
python_min_version = (3, 8, 0)

setuptools.setup(
    name="akitenkrad_blog_tools",
    version="0.0.1",
    author="akitenkrad",
    author_email="akitenkrad@gmail.com",
    packages=setuptools.find_packages(),
    package_data={
        "akitenkrad_blog_tools": [
            "keywords.json",
            "fonts/*.ttf",
            "templates/*.md",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
    + ["Programming Language :: Python :: 3.{}".format(i) for i in range(python_min_version[1], version_range_max)],
    long_description=long_description,
    install_requires=[
        "arxiv",
        "attrdict @ git+https://github.com/akitenkrad/attrdict",
        "click",
        "colorama",
        "googletrans==3.1.0a0",
        "nltk",
        "numpy",
        "openai",
        "pandas",
        "Pillow",
        "plotly",
        "progressbar",
        "py-cpuinfo",
        "pypdf",
        "python-dateutil",
        "pytz",
        "requests",
        "scikit-learn",
        "scipy",
        "slackweb",
        "sumeval",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "mypy",
            "flake8",
            "isort",
            "jupyterlab",
            "types-python-dateutil",
            "types-PyYAML",
            "types-requests",
            "typing-extensions",
        ]
    },
)
