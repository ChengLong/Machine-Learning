%function to compute the first m leading eigenvectors of matrix A 
function leading_eig = findLeadingEigV( A, m )
        [V,D]=eig(A);
   	[S,I]=sort(diag(D), 'descend'); %sort in descending order
        leading_eig = [];
        for i=1:m
	    leading_eig = [leading_eig,V(:,I(i))];
	endfor
endfunction

