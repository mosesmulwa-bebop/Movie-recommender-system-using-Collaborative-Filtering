function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X = num_movies  x num_features matrix of movie features
%        Theta = num_users  x num_features matrix of user features
%        Y = num_movies x num_users matrix of user ratings of movies
%        R = num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad = num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad = num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


##============Implementation Using for loop ====== -VERY SLOW
  
##  sum_of_stuff = 0;
##  for i = 1:num_movies,
##    for j = 1:num_users,
##      if(R(i,j) == 1),
##        myusertheta = Theta(j,:); %get row containing parameters for that user
##        myusertheta = myusertheta';  % get transponse to make it 100 * 1 vector
##        mymovieX = X(i,:); %this is a row containig all features for that movie -- 1 * 100 vecor
##        correcthypothesis = Y(i,j); %get the actual rating
##        % 1*100 * 100*1 = a scalar. Then get difference from correct hypothesis
##        difference = (mymovieX * myusertheta) - correcthypothesis; 
##        squareddifference = difference^2;
##        sum_of_stuff = sum_of_stuff + squareddifference;
##      endif
##    endfor
##  endfor
##
##  finalcost = (1/2) * sum_of_stuff;
  
  
 ### ==============VECTORIZED IMPLEMENTATION OF COST FUNCTION============================
   %X = nummovies * num_features
  newTheta = Theta'; %num_features * num_users
  predictionsmatrix = X * newTheta; % num_movies * num_users
  %%setting the elements of M to 0 only when the corresponding value in R is 0.
  zeroedPredictions = R .*predictionsmatrix;  % num_movies * num_users
  differencematrix = zeroedPredictions - Y;% num_movies * num_users
  squareddifferencematrix = differencematrix.^2;
  sum_of_everything = sum(sum(squareddifferencematrix)); %scalar
  totalcost = (1/2) * sum_of_everything;
  J =totalcost;
  
  ###==================GRADIENT====================
  ###X GRADIENT
  for i = 1:num_movies,
    idx= find(R(i,:)== 1);
    Thetatemp = Theta(idx,:);
    Ytemp = Y(i,idx);
    X_grad(i, :) = (X(i,:) * Thetatemp' - Ytemp)* Thetatemp; %1 movie * features
    singlerow = X(i,:);
    singlerow_reg = lambda* singlerow;
    X_grad(i,:) = X_grad(i,:) + singlerow_reg;
  endfor
 ###THETA GRADIENT
 for j = 1: num_users,
    idx= find(R(:,j)== 1); %Find all index where movies are rated
    Xtemp = X(idx,:); % get all rated movies * features
    Ytemp = Y(idx,j); % rated movies * 1 user
    
    Theta_grad(j,:) = ((Xtemp * Theta(j,:)') - Ytemp)' * Xtemp; %1 user * features
    singlerowtheta = Theta(j,:);
    singlerow_regtheta = lambda* singlerowtheta;
    Theta_grad(j,:) = Theta_grad(j,:) + singlerow_regtheta;
 endfor


   ###REGULARIZATION OF COST FUNCTION
   ##THETA
   
   thetasquared = Theta.^2;
   sumthetasquared = sum(sum(thetasquared,2));
   theta_reg = (lambda/2) * sumthetasquared;

   ##X
   Xsquared = X.^2;
   sumXsquared =  sum(sum(Xsquared));
   X_reg = (lambda/2) * sumXsquared;
   
   
   J = totalcost + theta_reg + X_reg;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
