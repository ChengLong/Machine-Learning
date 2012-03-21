function LdaImageDb =  constructImageDB(dbName, fold, rs, dataPath)
	if ( ~exist(dbName, "file") ) % if not exists
		disp('>>  no LdaImageDb.data. Now creating LdaImageDb...');
		folder = dir(dataPath);
		index=0;
		for i=1:40
		        folderName = folder(i+3).name;
			sub_folder_name = strcat(dataPath ,'/', folderName );
		        subdir = dir( sub_folder_name );
		        for j=1:10-fold
			    index++;
		            LdaImageDb(index).label = i;
		            img = imread(strcat( sub_folder_name, '/', subdir(j+2).name ));
		            % shrink img size. the original img size is too big for processing
		            res = shrinkImg(img, rs);
		            res = hisEqualize(res); % histogram equalization
		            %convert 2D image to 1D   
		            s=size(res);
		            LdaImageDb(index).image = reshape( res, s(1)*s(2), 1) ;
		        endfor
		endfor
		save -binary LdaImageDb.data LdaImageDb;
	else
		load("-binary", "LdaImageDb.data", "LdaImageDb");
	endif
endfunction
