function [L,rank1,piv,asvlues,error_bound] =   ...
          chol_part(loghyper,covfunc,X,tol,max_rank,ipiv)

% file  chol_part.m
% Useage:  
%    [L,rank1,piv] =  chol_part(loghyper,covfunc,X);
%       or
%    [L,rank1,piv,asvlues,error_bound] =  chol_part(loghyper,covfunc,X);
%       or
%    [L,rank1,piv,asvlues,error_bound] =  ...
%                 chol_part(loghyper,covfunc,X,tol,max_rank,ipiv);
%
% This function does a partial cholesky decomposition, with pivoting
%    or optionally without pivoting, of the kernel matrix K defined by
%      covfunc = a covariance function, created in the style of Rasmussen
%                and Williams,
%      loghyper = logarithm of hyperparameters and
%      X = the data matrix.  
%    For example, after defining covfunc, loghyper and X
%                  K = feval(covfunc,loghyper,X); or 
%                  K = feval(covfunc{:},loghyper,X);
%    With pivoting the columns and rows of the kernel 
%    are interchanged so that the more linearly independent columns and
%    rows appear first. The code requires calculating only a portion of
%    the kernel, not the entire matrix K, which can significantly reduce 
%    storage needs.
% Examples: covfunc='covSEiso'; loghyper=[0 0]; X = rand(10,5);
%           [L,rank1,piv]= chol_part(loghyper,covfunc,X);
%            X = X(piv,:); K = feval(covfunc,loghyper,X);
%              then K is completely factored
%                           or
%           covfunc='covSEiso'; loghyper=[0 0]; X = rand(10,5);
%           [L,rank1,piv]= chol_part(loghyper,covfunc,X,0,3);
%           X = X(piv,:); K = feval(covfunc,loghyper,X);
%              then L is a partial (3 columns) factorization of K
% Input:
%    loghyper is a (column) vector of log hyperparameters used in the
%          covariance function specified by covfunc.  Type help 
%          covFunctions for more details.  The number of hyperparameters
%          can be determined using feval.  For example
%                   covfunc ={'covSEiso'};  feval(covfunc{:})   or
%                   covfunc ='covSEiso'; feval(covfunc) 
%          return 2.  If we let loghyper = [ 0 0 ] then the hyperparameters
%          used by covSEiso are [1 1] since log([ 1 1 ] = [ 0 0 ].
%    covfunc  is the covariance function following Rasmussen and William
%          style.  See help covFunctions.  Example:
%               covfunc ='covSEiso' or covfunc ={'covSEiso'} 
%          Note: 
%          if covfunc is a covSum with the last term in the sum equal
%          to  covNoise (ex:  covfunc={'covSum',{'covSEiso','covNoise'}};)
%          the covNoise term is ignored in the Cholesky calculation 
%          (this is included since for some of Rasmussen and William's 
%           code uses a trick which involves an "extra" covNoise). 
%    X - the data matrix.  An n by d matrix.
%    tol (optional) - stopping tolerance.  If tol < 0, set tol =  n * eps 
%          where eps is relative machine precision 
%          (default: tol = -1 in order to use n* eps).
%          Let B = P' * K * P - L * L'.  
%          Stop factoring the matrix K when 
%               || B  ||_diag / || K ||_diag <= tol
%          Here || A ||_diag = max magnitude diagonal entry of A.
%          One can show for a symmetric positive definite matrix A that 
%                 || A ||_2 / n <= || A ||_diag <= || A ||_2
%          where ||  ||_2 is the usual matrix 2 norm.
%          Note: tol uses a RELATIVE criteria (|| B  ||_diag / || K ||_diag).
%    max_rank (optional) -  The maximum rank or maximum number of columns of L. 
%          If tol criteria is not met stop after factoring max_rank columns.
%          (default n).
%    ipiv (optional) =  do not pivot if ipiv = 0, pivot if ipiv is 1
%          ( the default is ipiv = 1 which uses pivoting).
% Output:
%    L - the n by rank1 partial cholesky factor (L is lower trapezoidal).
%    rank1 - the number of columns in the partial cholesky factor.
%        Also this is the rank of the approximation P * (L * L' ) * P' to K.
%        If rank1 = NaN the algorithm has failed. (This cannot happen for
%        ipiv = 1).
%    piv - An n vector that indicates the pivoting.
%        P' * K * P is factored where the permutation matrix P
%        has P ( i, piv(i) ) = 1.  The n by n matrix P is not 
%        explicitly formed.  
%        Note: if K is explicitly formed (which is not required) BEFORE
%        calling chol_part then K(piv,piv) = P' * K * P is factored.
%        If after calling chol_part we let X = X(piv) and  form
%        K then this K includes the pivoting and it is this K that is 
%        factored. In the comments below we will assume that K
%        refers to the initial K (before calling chol_part)
%    asvlues - rough estimates of the first rank1 + 1 singular values of K
%        but ONLY in the case that ipiv is 1 so that pivoting is done.
%        asvlues(1) = || K ||_diag is approx. the largest sing. value of K
%        asvlues(i) is approx. the ith singular value of K, i = 1, ...,rank1+ 1
%        asvlues(rank1 + 1) = || K  - P * L * L' * P'||_diag
%        Note for i = 1, ...., rank1, sqrt( asvlues(i) ) is approximately the
%            ith singular value of L and sqrt( asvlues(1) / asvlues(rank1) ) is 
%            approximately the condition number of L.
%        These values are only rough (order of magnitude) approximations since
%        they are calculated using the ||  ||_diag norm not the usual 2 norm
%        and since we are using the Cholesky factorization, with pivoting, 
%        not the singular value decomposition. 
%    error_bound - trace( E ) where E = K  - P * L * L' * P'.
%        In exact arithmetic (ignoring computer arith. errors) it follows that
%
%            asvlues(rank1 + 1 ) <= || E ||_2 <= || E ||_F <= error_bound
%
%        where || E ||_F is the Frobenius norm of E.
%
%    This program generates portions of K as needed.  This saves memory.  
%       The memory required for storing elements of K is at most 2*n not
%       n^2.  Also it saves work since only n * (rank1 + 1) elements of 
%       K are calculated.
%    Note that the covfunc specified must generate matrix elements
%       that come from a symmetric positive definite matrix.  Otherwise
%       the results of chol_part will be incorrect.
%
%    The memory requirement for this program is approximately
%               n * rank1 + n * d words (or numbers)
%       to store the n by rank1 matrix L and the n by d matrix X.
%       Also space is required for a few terms that require  O(n) storage.
%
%    The amount of work required by the program is, approximately, 
%                       n * rank1^2   flops
%       ( more precisely n * rank1^2 - 2 * rank1^3 / 3 flops) where a 
%       flop is either an addition, multiplication, subtraction or division.
%
%    Another Example:
%       covfunc ='covLINard'; n=4; loghyper = zeros(n,1); 
%       tol=0.1 ^ (n -1); ipiv = 1;
%       X=gallery('orthog',n)*diag(0.1 .^ (0:n-1));
%       [L,rank1,piv,asvlues,error_bound]=chol_part(loghyper,covfunc,X,tol);
%
%       For this example K = X * X' has eigenvalues ( = singular values since
%       K is spd) of 1, 1.e-2, 1.e-4, and 1.e-6.  The chol_part program
%       with tol = 1.e-3 produces a low rank approximation with rank = 2
%       and rough approximations (asvlues) of 3.63e-01  6.95e-03  1.44e-04 
%       for the first three singular values.  L is 4 by 2.  If after running
%       chol_part we set X = X(piv,:), K = X*X' and calculate
%                     E = K  - L * L' then 
%       || E ||_diag = 1.44e-4 <= || E ||_2 = 1.98e-4 <= error_bound = 2.00e-4


%    Written by L. Foster,  9-21-2006.  Based on a well known algorithm
%        ( for example see Golub and Van Loan, 3rd ed., p. 149)


[n,d]=size(X);

if ( ~exist('tol') ) 
   tol = -1;
end

tol_use = tol;
if ( tol < 0 ) tol_use = n * eps;  end ;

if ( exist('max_rank') ) 
   max_rank_use = min(n,max_rank);
else
   max_rank_use = n;
end

if ( ~exist('ipiv') ) 
   ipiv = 1;
end

% convert covfunc to a cell if needed
if ischar(covfunc)
    covfunc_use = cellstr(covfunc);
else
    covfunc_use = covfunc;
end

% if the covfunc_use cell array is a covSum, with last entry 'covNoise'
%    we remove the covNoise and to calculate the partial cholesky
%    factorization we use a covariance function without the last
%    covNoise.
if length(covfunc_use) == 2 & strcmp(covfunc_use(1), 'covSum') & ...
                                        strcmp(covfunc_use{2}(end), 'covNoise')
   covfunc_use = {covfunc_use{1},{covfunc_use{2}{1:end-1}}};
   loghyper_use = loghyper(1:end-1);
else
   covfunc_use = covfunc_use;
   loghyper_use = loghyper;
end


piv = (1:n)';                                                % initialize pivot
work = zeros(n,1);    % work will store sums of squares of entries in rows of L
L = zeros(n,max_rank_use);    % setting the initial size of the cholesky factor
                              % speeds up the algorithm significantly (compared
                                        % to initializing L with L = zeros(n,0)

% initial by calculating the diagonal entries of K
[K_diag, v] = feval(covfunc_use{:}, loghyper_use, zeros(0,d), X);   

diag_original = K_diag; 

max_diag  = max( K_diag );
asvlues(1) = max_diag;

for j = 1: max_rank_use + 1                 % loop over the columns / rows of K 

   if ( j > n )
      error_bound = 0;       % error_bound does not consider comp. arith. error
      break;            % >*>*>*>*>*> NOTE THIS BREAK, IT EXITS THE for j  LOOP
   end
   
   % find the largest diagonal entry of the unfactored part of the matrix
   [max_cur_diag, jpiv ] = max( diag_original(j:n) - work(j:n ) );
   asvlues(j) = max( max_cur_diag, 0 );
   % calculate error bound
   error_bound = norm(diag_original(j:n) - work(j:n ),1);
   jpiv = jpiv(1);                                     % needed in case of ties
   jpiv = j - 1 + jpiv ;
   if ( ipiv == 0 ) jpiv = j ; end                               % do not pivot
   
   % checking stopping criteria
   if ( max_cur_diag <= tol_use * max_diag | j > max_rank_use )
      rank1 = j - 1;
      break;            % >*>*>*>*>*> NOTE THIS BREAK, IT EXITS THE for j  LOOP
   end
   
   if ( jpiv ~= j )
       
       % switch entries in piv, work, diag_original, and L and X
       
       piv([j jpiv]) = piv([ jpiv j ] );
       work([j jpiv]) = work([ jpiv j ] );
       diag_original([j jpiv]) = diag_original([ jpiv j ] );
       
       temp = X(j,:);              % note: switching rows of X will, in effect, 
       X(j,:) = X(jpiv,:);                    % also switch rows / columns of K
       X(jpiv,:) = temp;
       
       temp = L(j,:);
       L(j,:) = L(jpiv,:);
       L(jpiv,:) = temp;
       
   end
   
   cur_diag_jj = diag_original(j) - work(j);
   if ( cur_diag_jj > 0 )
      L(j,j) = sqrt( cur_diag_jj );            % calculate next diag entry of L
   else
      L(j,j) = sqrt( tol_use * max_diag );
      disp([ ' chol_part may be inaccurate. ', ...
            ' Try setting ipiv = 1 and rerunning. '])
      
      % If we are in this case then due to computer arithmetic errors
      % cur_diag_jj < 0 even though in theory cur_diag_jj >= 0.  So that
      % the algorithm does not fail we replace cur_diag_jj by 
      % tol_use * max_diag.  This corresponds to perturbing a diagonal
      % entry of the original K by a small amount. 
      % Remark: this alternative is only possible with ipiv = 0
      
      % Older altenative code:
      %disp([ ' chol_part may be inaccurate. ', ...
      %      ' Try setting ipiv = 1 and rerunning. '])
      %rank1 = j-1;
      %break;
   end
   
   L(1:j-1,j) = zeros(j-1,1);
   
   % calculate part of column j of K   
   j_column = j;
   i_row = j + 1 ;
   
   % Calculate elements in rows i_row to n of column j_column of K
   [v, K_col] = feval(covfunc_use{:},loghyper_use,X(i_row:n,:), X(j_column,:));   
   
   % calculate rest of column j of L
   L(j+1:n, j) = ( K_col - L(j+1:n,1:j-1)*( L(j,1:j-1)' ) ) / L(j,j);
   % update the work vector
   work(j+1:n) = work(j+1:n) + L( j+1:n,j ).^2 ;
   
   rank1 = j;
   
end

if ( rank1 < max_rank_use ) 
   L = L(:,1:rank1);
end