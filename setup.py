#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['matplotlib>=3.7.1', 
                'numpy>=1.21.5', 
                'pandas>=2.0.2', 
                'tqdm>=4.63.0']

test_requirements = ['pytest>=3', ]

setup(
    author="Brecht Wuyts ",
    author_email='brecht.wuyts@kuleuven.be',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="The DyLoPro Python Library is a visual analytics tool that allows \
Process Mining (PM) practitioners to efficiently and comprehensively \
explore the dynamics in event logs over time, prior to applying PM \
techniques. These comprehensive exploration capabilities are provided \
by extensive set of plotting functionalities, visualizing the dynamics \
over time from different process perspectives.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='DyLoPro',
    name='DyLoPro',
    packages=find_packages(include=['DyLoPro', 'DyLoPro.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/BrechtWts/DyLoPro',
    version='0.1.0',
    zip_safe=False,
)
