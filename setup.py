from setuptools import setup, find_packages

setup(
    name='vlm_inference',
    version='0.1.01',
    packages=find_packages(),
    install_requires=[
    'openai==1.78.1',
    'opencv-python==4.11.0.86',
    'pyyaml',  # Si no especificas versión, solo el nombre del paquete
    'python-dotenv==0.9.1,<0.11', # El paquete 'dotenv' se llama 'python-dotenv' en PyPI
    'Pillow==9.0.1' # El paquete 'pillow' se llama 'Pillow' en PyPI (sensible a mayúsculas)
    ],
    include_package_data=True,
    author='Edison Bejarano',
    description='Vision language models for robotics',
    license='MIT',
    url='https://gitlab.iri.upc.edu/mobile_robotics/moonshot_project/vlm/vlm_inference',
)