function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


z = X * theta; % (m x 2) * (2 x 1) = (1 x m)
hypothesis = sigmoid(z);
variance = (hypothesis - y).^2; % ( (m x 1) - (m x 1) ) ^2
J =  (2*m)^-1  * sum(variance); % (1 x 1) * ( (1 x m) * (m x 2) = (1 x 2) );

grad = variance; % To-Do: double check this!!

% =============================================================

end
