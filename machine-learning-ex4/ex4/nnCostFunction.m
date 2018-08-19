function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Compute h_theta(X) by using the Feedforward algorithm.
% Add bias to input layer
a1 = [ones(m, 1), X]; % Size = m x (input_layer_size + 1);

z2 = Theta1 * a1';
a2 = sigmoid(z2); % Size = hidden_layer_size x m;
% Add bias to hidden layer
a2 = [ones(1, m) ; a2]; % Size = (hidden_layer_size + 1) x m;

% Compute h_theta(x) aka. a3
z3 = Theta2 * a2;
a3 = sigmoid(z3)'; % Size =  m x num_labels

% Now that we have h_theta(X) we can compute the cost aka J(Theta)
y_bin = zeros(m, num_labels);

for i = 1:m
  % Find all the index of where (y == K)
  % We need to turn y from decimal values to "binary" values.  % 3 = [0 0 1 0 0 0 0 0 0 0]
  K = y(i, 1);
  y_bin(i, K) =  1;
end
%
% for i = 1:m
%   if mod(i, 250) == 0
%     fprintf('Label of item m = %d = [%d %d %d %d %d %d %d %d %d %d]\n', [y(i, 1), y_bin(i, :)]);
%   end
%   for K = 1:num_labels
%     cost = (-y_bin(i,K) * log(a3(i,K))) - ((1-y_bin)(i,K) * log(1-a3(i, K)));
%     J += cost;
%
%     if mod(i, 250) == 0
%       fprintf('Cost for label %d = %f\n', [K, cost]);
%     end
%   end
%   if mod(i, 250) == 0
%     fprintf('i = %f\n', i);
%     fprintf('Cost = %f\n', J);
%     fprintf('y_bin = [[%d %d %d %d %d %d %d %d %d %d]\n', y_bin(i,:));
%     fprintf('h(X) = [[%d %d %d %d %d %d %d %d %d %d]\n', a3(i,:));
%     fprintf('Log(h(X)) = [%f %f %f %f %f %f %f %f %f %f]\n', log(a3(i,:)));
%     fprintf('\n');
%   end
% end

for K = 1:num_labels
  pos = find(y == K);
  % We need to turn y from decimal values to "binary" values.  % 3 = [0 0 1 0 0 0 0 0 0 0]
  % y_k = zeros(length(pos), num_labels);
  % y_k(:, K) = 1;
  a3_k = a3(pos, :);
  y_k = (y_bin(pos, :))';

  % fprintf('Value of y for label %d = [%d %d %d %d %d %d %d %d %d %d]\n', [K, y_bin(1, :)]);
  % fprintf('Size of y_k %dx%d\n', size(y_k));
  % fprintf('Size of a3 %dx%d\n', size(a3_k));
  % fprintf('m = %d\n', m);
  % fprintf('Cost =  %f\n', J);
  % fprintf('Added cost = %f\n\n', trace( y_k' * (log(a3_k) ) + ( (1-y_k)' * log(1-a3_k))));

  J += trace( -y_k * log(a3_k) - (1-y_k) * log(1-a3_k) );

end

J = J/m;
J += lambda/(2*m) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));


% Calculate gradients
b3 = a3 - y_bin; % beta1
b2 = (b3 * Theta2(:, 2:end)) .* sigmoidGradient(z2'); % beta2

% Implement Backpropagation
d1 = (a1' * b2)';
d2 = (a2 * b3)';

Theta1_grad = d1/m;
Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);

Theta2_grad = d2/m ;
Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end);








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:)];


end
