from setuptools import setup, find_packages

setup(
    name="alpha-architecture-agent",
    version="0.1.0",
    description="AI Agent-based Stock Prediction Architecture Explorer",
    author="Alpha Architecture Team",
    author_email="team@alphaarchitecture.ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "alpha-agent=core.cli:main",
            "alpha-train=core.train:main",
            "alpha-backtest=core.backtest:main",
        ],
    },
}