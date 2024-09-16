from setuptools import setup, find_packages


setup(
    name="tridf",
    version="0.0.1",
    description="DataFrame powered by Triton kernels",
    author="Jiashen Cao",
    author_email="caojiashen24@gmail.com",
    url="github.com/jiashenC/triDF",
    packages=find_packages(),
    install_requires=[
        "torch",
        "triton",
    ],
)
