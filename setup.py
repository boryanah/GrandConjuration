from setuptools import find_packages, setup

setup(
    name="galshcl_like",
    version="0.0",
    description="Toy galaxy, shear C_ell likelihood",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "cobaya>=3.0",
        "sacc>=0.4.2",
    ],
    package_data={"galshcl_like": ["GalShClLike.yaml"]},
)
