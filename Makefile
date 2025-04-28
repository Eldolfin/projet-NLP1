demo:
	mkdir -p .cache
	python src/main.py

console:
	python src/console.py

watch:
	git ls-files | entr -c make demo
