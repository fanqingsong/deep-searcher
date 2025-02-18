from setuptools import setup, find_packages

setup(
    name='deepsearcher',
    version='0.0.1',
    py_modules=['deepsearcher'],
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        'argparse',
        'firecrawl-py',
        'langchain_text_splitters',
        'pdfplumber',
        'pymilvus[model]',
        'openai',
        'numpy',
        'tqdm',
        'termcolor',
        'fastapi',
        'uvicorn',
        'pydantic-settings'
    ],
    entry_points={
        'console_scripts': ['deepsearcher=deepsearcher.cli:main'],
    },
    description='None',
    author='Cheney Zhang',
    author_email='277584121@qq.com',
    url="https://github.com/zilliztech/deep-searcher",

)
