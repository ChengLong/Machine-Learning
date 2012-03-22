#!/bin/bash

for fold in 1 2 3 4 5 6
do
	octave -q FaceRec_LDA.m 5 $fold 2
	rm LdaImageDb.data faceDb.data W_optimal.data;
done	
