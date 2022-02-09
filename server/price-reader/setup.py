from setuptools import find_packages, setup

packages = [p for p in find_packages() if "tests" not in p]

setup(
    name="price_reader",
    version="0.1.0",
    description="Price reader library",
    packages=packages,
    python_requires=">=3.7, <4",
    include_package_data=True,
    zip_safe=False,
)
