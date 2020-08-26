rm -rf ./build
rm -rf ./dist
rm -rf ./singa_auto.egg-info
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*  --u naili --p  singaauto
rm -rf ./build
rm -rf ./dist
rm -rf ./singa_auto.egg-info
