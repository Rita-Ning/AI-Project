"""
File: interactive.py
Name: Rita Tang
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

from submission import *


def main():
	with open('weights','r', encoding='utf-8') as f:
		weights = {}
		for line in f:
			data_lst = line.split()
			weights[data_lst[0]] = float(data_lst[1])
	interactivePrompt(extractWordFeatures, weights)



if __name__ == '__main__':
	main()