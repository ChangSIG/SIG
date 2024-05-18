from setuptools import setup, find_packages

setup(
	name='sig',
	version='0.0.1',
	description='Python package to perform network based stratification of binary somatic mutations as described in SIGv1.',
	url='https://github.com/ChangSIG/SIG.git',
	author='Zhang',
	author_email='zcc@hyhzcc.xin',
	license='MIT',
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'Topic :: Software Development :: Build Tools',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3.7'
	],
	packages=find_packages(exclude=['os', 'random', 'time']),
	install_requires=[
        'lifelines>=0.9.1',
        'networkx>=2.0',
        'numpy>=1.11.0',
        'matplotlib>=1.5.1',
        'pandas>=0.19.0',
        'scipy>=0.17.0',
        'scikit-learn>=0.17.1',
        'seaborn>=0.7.1']
)
