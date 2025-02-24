import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
long_description = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="mikasa-base",
    version="0.1.0",
    description="MIKASA‑Base: A Unified Benchmark for Memory‑Intensive Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Egor Cherepanov, Nikita Kachaev, Alexey K. Kovalev, Aleksandr I. Panov",
    author_email="contact@example.com",
    url="https://sites.google.com/view/memorybenchrobots/",
    project_urls={
        "Source": "https://github.com/CognitiveAISystems/MIKASA-Base",  
        "Site": "https://sites.google.com/view/memorybenchrobots/"
    },
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "absl-py==2.1.0",
        "dm_control==1.0.27",
        "dm_env==1.6",
        "dm_tree==0.1.8",
        "gymnasium==0.29.1",
        "gym-minigrid==1.2.2",
        "numpy>=1.26.0,<2.1.0",
        "scipy==1.13.1",
        "tensorflow==2.18.0",
        "torch==2.4.0",
        "stable-baselines3==2.5.0",
        "ray==2.40.0",
        "mujoco==3.2.7"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries"
    ],
    keywords=[
        "reinforcement-learning",
        "memory",
        "benchmark",
        "MIKASA",
        "POMDP",
        "gymnasium"
    ],
)
