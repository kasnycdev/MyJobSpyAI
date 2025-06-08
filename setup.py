from setuptools import setup, find_packages

setup(
    name="myjobspyai",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # List your package dependencies here
    ],
    python_requires=">=3.8",
    author="MyJobSpyAI Team",
    author_email="your.email@example.com",
    description="AI-powered job search and analysis tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kasnycdev/MyJobSpyAI",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
