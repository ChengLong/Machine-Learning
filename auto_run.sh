#!/bin/bash

for shrink in 2 3 4
do
	for fold in 1 2 3 4 5
	do
		for k in 1 2 3 4 5
		do
			octave -q lda.m $k $fold $shrink
       		done
		rm LdaImageDb.data faceDb.data W_optimal.data;
	done	
done
