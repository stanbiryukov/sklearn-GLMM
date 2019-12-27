from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("sklearn-GLMM/version.py") as f:
    exec(f.read())

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name ='sklearn-GLMM',
    version = __version__,
    author ='Stan Biryukov',
    author_email ='stan0625@uw.com',
    url = 'git@github.com:stanbiryukov/sklearn-GLMM.git',
    install_requires = requirements,
    package_data = {'sklearn-GLMM':['resources/*']},
    packages = find_packages(exclude=['sklearn-GLMM/tests']),
    license = 'MIT',
    description='sklearn-GLMM: scikit-learn wrapper for generalized linear mixed model methods in R',
    long_description= "sklearn-GLMM makes it easy to interface with GLMM libraries in R. It was specifically designed to work with BRMS and LME4 but is also flexible enough to work with other stats libraries",
    keywords = ['statistics','multi-level-modeling','regression', 'analysis', 'bayesian', 'machine-learning', 'scikit-learn'],
    classifiers = [
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)