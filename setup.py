from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='nmf_emma',
	version='0.0.1',
	description='Non-negative matrix factorization end-member mixing analysis',
	long_description=readme(),
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Science/Research',
		'License :: Free for non-commercial use',
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 2',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 2.7',
		'Topic :: Scientific/Engineering'
	],
	url='https://github.com/FluvialSeds/nmf_emma',
	download_url='https://github.com/FluvialSeds/nmf_emma/tarball/0.0.1',
	keywords=[
		'geochemistry',
		'carbon cycle',
		'weathering',
		'mixing models',
	],
	author='Jordon D. Hemingway',
	author_email='jordon.hemingway@erdw.ethz.ch',
	license='GNU GPL Version 3',
	packages=['nmf_emma'],
	install_requires=[
		'matplotlib',
		'numpy',
		'pandas',
		'scipy',
		'sklearn',
		'tqdm',
	],
	# test_suite='nose.collector',
	# tests_require=['nose'],
	include_package_data=True,
	# zip_safe=False
	)