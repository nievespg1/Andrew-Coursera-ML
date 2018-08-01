function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


hypothesis = X * theta; % (m x 2) * (2 x 1) = (1 x m)
variance = (hypothesis - y).^2; % ( (m x 1) - (m x 1) ) ^2
J =  (2*m)^-1  * sum(variance); % (1 x 1) * ( (1 x m) * (m x 2) = (1 x 2) );


% =========================================================================

end
