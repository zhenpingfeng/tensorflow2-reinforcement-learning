from setuptools import setup

setup(
    name="forexrl",
    py_modules="forexrl",
    version="1.0.0",
    install_requires=[
        "numpy",
        "ipython",
        "tensorflow>=2.0",
        "pandas",
        "ta"
    ],
    description="forex trading by deep reinforcement learning"
)
