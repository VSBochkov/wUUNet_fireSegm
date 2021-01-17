from os.path import dirname, join

from setuptools import setup, find_packages
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='wUUNet_fireSegm',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/VSBochkov/wUUNet_fireSegm',
    license='GNU GPLv3',
    install_requires=install_requires,
    author='Vladimir Bochkov',
    author_email='vladimir2612@bk.ru',
    description='The pytorch implementation of wUUNet model applicable for multiclass fire-segmentation',
    long_description=open(join(dirname(__file__), 'README.md')).read()
)
