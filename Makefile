# data/training.1600000.processed.noemoticon.csv:
# 	cd data
# 	unzip dataset.zip
	
demo:
	python main.py

watch:
	git ls-files | entr -c make demo
