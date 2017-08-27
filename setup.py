from setuptools import setup, find_packages

setup(
    name='wonders_calculator',
    version='0.1',
    description="Automatic points calculation at board game 7 wonders",
    url="",
    author='Yoann Benoit',
    author_email='',
    license='new BSD',
    packages=find_packages(),
    install_requires=['tensorflow'],
    tests_require=[],
    scripts=[],
    py_modules=["wonders_calculator"],
    include_package_data=True,
    zip_safe=False
)