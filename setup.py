try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A collection of metrics and loss functions in Theano.',
    'author': 'Jonathan Strong',
    'url': 'https://github.com/jonathanstrong/tmetrics',
    'download_url': 'https://github.com/jonathanstrong/tmetrics',
    'author_email': 'jonathan.strong@gmail.com',
    'version': '0.0.1',
    'install_requires': ['nose', 'scikit-learn', 'theano'],
    'packages': ['tmetrics'],
    'scripts': [],
    'name': 'tmetrics'
}

setup(**config)
