from setuptools import find_packages, setup

setup(
    name='mcgenerator',
    version='0.0.1',
    author='Claudius Joel',
    author_email='otiiclaudius@gmail.com',
    packages=find_packages(),
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
)