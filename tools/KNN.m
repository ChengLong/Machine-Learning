
function [correct, totalTesting] = KNN(W_optimal, faceDb, fold, rs, kValue, dataPath)
	correct = 0;
	totalTesting = 0;

	folder = dir(dataPath);
	for i=1:40
	    folderName = folder(i+3).name;
            sub_folder_name = strcat(dataPath ,'/', folderName );
	    subdir = dir( sub_folder_name );
	    for j=11-fold:10
		totalTesting++;
		testing.label = i;
		img = imread( strcat(sub_folder_name , '/', subdir(j+2).name ) );

		% shrink img size accroding to requirement
		res = shrinkImg(img, rs);
		res = hisEqualize(res);
		s = size(res);
		testing.image = W_optimal * reshape(res, s(1)*s(2), 1);

		% knn
		% 1-distance 2-label
		candidates = [];
		for k=1:length(faceDb),
		    % calculate norm
		    candidates(k,1) = norm(testing.image - faceDb(k).image, 2);
		    candidates(k,2) = faceDb(k).label;
		endfor

		candidates = sortrows(candidates, 1);
		maxLabel = 40;
		labels = zeros(1, maxLabel, "uint8");
		for k=1:kValue
		    labels( candidates(k,2) )++;%vote
		endfor
		res = 0;
		maxVotes=0;
		for k=1:kValue % check at most kValue position in the row vector
		    if labels(candidates(k,2)) > maxVotes,
		        maxVotes=labels(candidates(k,2));
		        res = candidates(k,2);
		    endif
		endfor
		%printf('>>  testing %3i **  predicted: %4i(votes=%i) real: %4i\n', totalTesting, res, maxVotes, testing.label);

		if res == testing.label,
		    correct++;
		endif
	    endfor
	endfor
endfunction
