#!/bin/bash

models=(gpt2-xl)

for mname in "${models[@]}"
do
	python3 baseline.py --model-name $mname --batch-size 8 --window-size 20
	python3 baseline.py --model-name $mname --batch-size 8 --window-size 20 --remove-format-chars
	python3 baseline.py --model-name $mname --batch-size 8 --window-size 20 --remove-punc-spacing 
	python3 baseline.py --model-name $mname --batch-size 8 --window-size 20 --remove-format-chars --remove-punc-spacing 

done
