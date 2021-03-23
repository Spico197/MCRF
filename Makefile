build: clean
	python3 setup.py sdist bdist_wheel

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name '*.pyc' -type f -delete
	find . -name '__pycache__' -type d -delete

test_upload:
	python3 -m twine upload --repository testpypi dist/*

upload:
	python3 -m twine upload --repository pypi dist/*

test:
	python -m unittest discover -v tests
	flake8
