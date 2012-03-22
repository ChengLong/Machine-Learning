
function W_optimal = LDA(fold, LdaImageDb)
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
		        S_within_class = S_within_class + cov(each_class);
		endfor
		%calculate between-class scatter using mean_of_all_class
		S_between_class = zeros(dim_of_img);
		%since each class has exactly the same # training images, the mean of the mean of each class will be the mean of all 
		%training data
		S_between_class =  num_of_training * cov(mean_of_all_class);

		% it can be shown that the optimal projection matrix W_optimal is the one whose columns are the first (C-1) leading 
		%eigenvectors of inv(S_within_class)*S_between_class
		S_inv_w_b = pinv(S_within_class) * S_between_class;

		%get the first C-1 = 39 eigenvectors 
		W_optimal = findLeadingEigV(S_inv_w_b, 39);
		W_optimal = W_optimal'; % to make it easy for projection

		save -binary W_optimal.data W_optimal;
	endif
endfunction
