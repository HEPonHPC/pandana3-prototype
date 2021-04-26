import setuptools


with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="pandana3", packages=["pandana3"], install_requires=install_requires
)
