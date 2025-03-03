data/training.1600000.processed.noemoticon.csv:
	cd data
	unzip dataset.zip
	
demo: data/training.1600000.processed.noemoticon.csv
	python main.py
