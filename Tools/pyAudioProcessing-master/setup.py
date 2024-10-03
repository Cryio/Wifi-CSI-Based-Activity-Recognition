import os
import codecs
from setuptools import setup, find_packages

PROJECT = os.path.abspath(os.path.dirname(__file__))
REQUIREMENTS_PATH = "requirements/requirements.txt"
# The text of the README file
with open("README.md", "r") as f:
    long_description = f.read()

def read(path):
    with codecs.open(os.path.join(PROJECT, path), "rb", "utf-8") as f:
        return f.read()

def get_requirements(path=REQUIREMENTS_PATH):
    """
    Returns a list of requirements defined by REQUIREMENTS_PATH.
    """
    requirements = []
    for line in read(path).splitlines():
        requirements.append(line.strip())
    return requirements

setup(
   name='pyAudioProcessing',
   version='1.1.13',
   description='Audio processing-feature extraction and building machine learning models from audio data.',
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Jyotika Singh',
   packages=find_packages(where=PROJECT),
   package_data={'': ['svm_clf', 'svm_clf.arff', 'svm_clfMEANS']},
   include_package_data=True,
   url="https://github.com/jsingh811/pyAudioProcessing",
   install_requires=get_requirements(),
   python_requires='>=3.6, <4',
   py_modules=["pyAudioProcessing"],
   classifiers=[
       "Programming Language :: Python :: 3",
       "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
       "Operating System :: OS Independent",
   ]
)
