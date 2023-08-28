

build-wheel:
	python setup.py bdist_wheel
	echo value is "$PYPI_TOKEN"
	twine upload -r pypi dist/* -u __token__ -p "$PYPI_TOKEN"


activate-dev-venv:
	pip install -e .

remove-pip-packages:
	pip uninstall -y -r <(pip freeze)


documentation:
	sphinx-apidoc -o docs src/
	cd docs
	make html clean

