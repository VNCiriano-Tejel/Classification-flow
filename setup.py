from setuptools import setup, find_packages

setup(
    name = "flow_class_vir",
    version = "1.0",
	author='Virginia Ciriano',
    author_email='virginia.ciriano@gmail.com',
	license='LICENSE.md',
	description='Unsupervised learning for feature extraction in a flow classifier',
    long_description=open('README.md').read(),
    packages=[ 'flow_class_vir','flow_class_vir.data_cleaning', 'flow_class_vir.Plot_color_map', 'flow_class_vir.Autoencoder'],
    )
