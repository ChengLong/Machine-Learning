#!/bin/bash

for shrink in 2 3 4
do
	for fold in 1 2 3 4 5 6 7
	do
		octave -q lda.m $((10 - $fold)) $fold $shrink
		rm LdaImageDb.data faceDb.data W_optimal.data;
	done	
done
