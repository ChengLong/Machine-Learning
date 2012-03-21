function faceDb = constructFaceDB(LdaImageDb, W_optimal)
	if (exist('faceDb.data', "file") ),
		load ("-binary", "faceDb.data", "faceDb");
	else
		for i=1:length(LdaImageDb),
		    faceDb(i).label = LdaImageDb(i).label;
		    faceDb(i).image = W_optimal * LdaImageDb(i).image;
		endfor
		save -binary faceDb.data faceDb;
	endif
endfunction
