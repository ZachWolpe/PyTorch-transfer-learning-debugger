from setuptools import setup, find_packages

setup(
    name="pytorch-transfer-learning-debugger",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core ML packages
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        
        # Data processing & visualization
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black",
            "flake8",
            "mypy",
        ],
    },
    author="Zach Colin Wolpe",
    author_email="zachcolinwolpe@gmail.com",
    description="A debugger for PyTorch transfer-learning & fine-tuning jobs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pytorch-transfer-learning-debugger",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)