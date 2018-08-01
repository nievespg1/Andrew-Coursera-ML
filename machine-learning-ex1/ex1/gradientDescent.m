function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    hypothesis = X * theta; % (m x 2) * (2 x 1) = (1 x m)
    variance = hypothesis - y; % (m x 1) - (m x 1) = (m x 1)
    update_value = (alpha/m) * (variance' * X); % (1 x 1) * ( (1 x m) * (m x 2) = (1 x 2) );
    theta = theta - update_value'; % (2 x 1) - (2 x 1) = (2 x 1)


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
end

end
