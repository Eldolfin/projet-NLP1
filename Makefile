# data/training.1600000.processed.noemoticon.csv:
# 	cd data
# 	unzip dataset.zip
	
demo:
	python src/main.py | lolcat

watch:
	git ls-files | entr -c make demo
