from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='baseLess',
    version='0.0.1',
    packages=['baseLess'],
    install_requires=[
        'matplotlib==3.5.1',  # avoid issues with heatmaps
        'Cython==0.29.30',
        'tensorflow-gpu==2.8.1',
        'pandas==1.4.2',
        'bokeh==2.4.3',
        'snakemake==7.8.1',
        'h5py==3.6.0',
        'zodb==5.7.0',
        'hyperopt==0.2.7',
        'seaborn==0.11.2',
        'numpy==1.22.4',
        'tables==3.7.0',
        'scikit-learn==1.1.1',
        'fastparquet==0.8.1'
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.10',
    author='Carlos de Lannoy',
    author_email='cvdelannoy@gmail.com',
    description='Detect sequences in MinION squiggles, no basecalling required',
    long_description=readme(),
    license='MIT',
    keywords='nanopore sequencing,MinION,sequence detection',
    url='https://github.com/cvdelannoy/baseLess',
    entry_points={
        'console_scripts': [
            'baseLess = baseLess.__main__:main'
        ]
    },
    include_package_data=True
)
