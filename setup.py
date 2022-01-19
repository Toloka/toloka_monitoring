from setuptools import setup, find_packages

setup(
    name='toloka_monitoring',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'toloka_monitoring = toloka_monitoring.__main__:main',
        ]
    },
    packages=find_packages(),
    long_description=__doc__,
    include_package_data=True,
)
