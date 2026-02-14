#!/bin/bash

input="lista-censo-edos.txt"
while read -r line
do
        echo "$line"        
	curl "$line" -o tmp.zip
	mkdir tmp && mv tmp.zip tmp/ && unzip tmp/tmp.zip  -d tmp/
        mv tmp/conjunto_de_datos/*.csv csvs/
        rm -rf tmp 
done < "$input"
