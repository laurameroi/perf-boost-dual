try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='ren-controllers',
    version='1.0.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=[
        'plants',
        'loss_functions',
        'controllers',
        'plants.robots',
    ],
)
