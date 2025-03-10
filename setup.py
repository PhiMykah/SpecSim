from setuptools import setup,find_packages

setup(name='specsim',
    version='0.6.0',
    packages=find_packages(), 
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'specsim = specsim.main:main',
        ]
    },
    author='Micah Smith',
    author_email='mykahsmith21@gmail.com',
    description='Nuclear Magnetic Resonance (NMR) spectrum signal simulator'
)
