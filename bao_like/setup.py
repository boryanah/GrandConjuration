from setuptools import find_packages, setup


setup(
    name="bao_like",
    version="0.0",
    description="Additional BAO likelihoods",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "cobaya>=3.0",
    ],
    package_data={"bao_like": ["bao.yaml"]}, # I have no idea what the yaml file is for
)
