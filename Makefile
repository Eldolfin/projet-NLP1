demo:
	mkdir -p .cache
	python src/main.py

watch:
	git ls-files | entr -c make demo
