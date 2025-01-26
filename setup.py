from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="actrix",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': ['actrix=actrix.__main__:main']
    },
    install_requires=required + ['tabulate']
)