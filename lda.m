
% shrink image size according to requirment
function shrinkedImg = shrinkImg (img, rs)
	[r,c]=size(img);
	shrinkedImg = img(1:rs:r , 1:rs:c );
endfunction

% function to compute the square of euclidean distance between r1 and r2
function res = SquareEuc(r1, r2)
    len = length(r1);
    res = 0;
    for i=1:len
        res += (r1(i)-r2(i))^2;
    endfor
endfunction

%function to compute the first m leading eigenvectors of a matrix
function leading_eig = findLeadingEigV( A, m )
	k=size(A,1); % assume A is square
	[V,D]=eig(A); 
	[S,I]=sort(diag(D)); 
	leading_eig = [];
	for i=1:m
		leading_eig = [leading_eig,V(:,I(k-i+1))];
	endfor
endfunction

% 0. Initialization

pkg load image;
pkg load strings;

args = argv();

if nargin<1,
    disp('Format: k-Value [fold=1] [image scaling=1] [PCA d]');
    exit(0);
endif

kValue = str2num(args{1});
maxLabel = 0;

fold=1;
if nargin >1,
    fold = str2num(args{2});
endif

rs = 1;
if nargin>=3
    rs = str2num(args{3});
endif

printf('\n\n*************************\nExperiment: kValue=%d, fold=%i, shrink=%i\n', kValue, fold, rs);
 
% 2. Construct LdaImageDB: Read Images

disp('1. Constructing LdaImageDb...');
if ( ~exist('LdaImageDb.data', "file") ) % if not exists
	disp('>>  no LdaImageDb.data. Now creating LdaImageDb...');
	cd dataset;
	folder = dir('.');
	index=0;
	for i=1:40
		folderName = folder(i+3).name;
		subdir = dir( folderName );
		for j=1:10-fold
		    index++;
		    LdaImageDb(index).label = i;
		    if i>maxLabel,
			maxLabel=i;
		    endif
		    img = imread( strcat( folderName, '/', subdir(j+2).name ) );
		    
		    % shrink img size. the original img size is too big for processing
		    res = shrinkImg(img, rs);
		    %convert 2D image to 1D   
		    s=size(res);
		    LdaImageDb(index).image = reshape( res, s(1)*s(2), 1) ;
		endfor
	endfor
	cd ..;
	save -binary LdaImageDb.data LdaImageDb;
else
	load("-binary", "LdaImageDb.data", "LdaImageDb");
endif

% 2. Dimensionality Reduction: LDA
disp('2. Dimensinality Reduction...')

% calculate projection matrix W
if (exist('W_optimal.data', "file") ),
	load ("-binary", "W_optimal.data","W_optimal");
else %compute it and save for next time 
	% calculate within class scatter Sw = Sigma(Si)
	num_of_training=10-fold;
	dim_of_img = rows(LdaImageDb(1).image);
	S_within_class = zeros(dim_of_img);
	mean_of_all_class = []; % each row is the mean of a class
	for i=1:40
		% extract all training data for each class 
		each_class=[];
		base = num_of_training * (i-1);
		for j=1:num_of_training
			each_class= [each_class ; transpose(LdaImageDb(base+j).image)];
		endfor
		mean_of_all_class = [mean_of_all_class; mean(each_class)];

		%calculate covariance matrix, 
		S_within_class = S_within_class + cov(double(each_class));
	endfor

	%calculate between-class scatter using mean_of_all_class
	S_between_class = zeros(dim_of_img);
	%since each class has exactly the same # training images, the mean of the mean of each class will be the mean of all training data
	S_between_class =  num_of_training * cov(mean_of_all_class);

	% it can be shown that the optimal projection matrix W_optimal is the one whose columns are the first (C-1) leading eigenvectors of 
	%inv(S_within_class)*S_between_class
	S_inv_w_b = inv(S_within_class) * S_between_class;

	%get the first C-1 = 39 eigenvectors 
	W_optimal = findLeadingEigV(S_inv_w_b, 39);
	W_optimal = W_optimal'; % to make it easy for projection

	save -binary W_optimal.data W_optimal;
endif

% 3. Construct FaceDB: Project Data to low-dimensional space
disp('3. Constructing FaceDB');
if (exist('faceDb.data', "file") ),
	load ("-binary", "faceDb.data", "faceDb");
else
	for i=1:length(LdaImageDb),
	    faceDb(i).label = LdaImageDb(i).label;
	    faceDb(i).image = W_optimal * double(LdaImageDb(i).image);
	endfor
	save -binary faceDb.data faceDb;
endif

% 4. Classification: KNN Nearest Neighbor
disp('4. Classifying using KNN');
correct = 0;
totalTesting = 0;

cd dataset;
folder = dir('.');
for i=1:40
    folderName = folder(i+3).name;
    subdir = dir( folderName );
    for j=11-fold:10
        totalTesting++;
        testing.label = i;
        img = imread( strcat( folderName, '/', subdir(j+2).name ) );
        s = size(img);
        
        % shrink img size accroding to requirement
	res = shrinkImg(img, rs);
        s = size(res);
        testing.image = W_optimal * double(reshape(res, s(1)*s(2), 1));

	% knn
        % 1-distance 2-label
        candidates = [];
        for k=1:length(faceDb),
            candidates(k,1) = SquareEuc(testing.image, faceDb(k).image);
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
cd ..;

disp(sprintf('accuracy = %i / %i = %f%%', correct, totalTesting, correct/totalTesting*100) );
