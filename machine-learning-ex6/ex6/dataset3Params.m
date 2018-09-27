function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
Sig_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

current_error = 1;
C = 1; sigma = 1;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
for i = 1:length(C_vec)
  tmp_c = C_vec(i);
  for j = 1:length(Sig_vec)
    tmp_sig = Sig_vec(j);
    % Train model using C_vec[i] and Sig_vec[j];
    % Evaluate performance of model using cross validation dataset
    model = svmTrain(X, y, tmp_c, @(x1, x2) gaussianKernel(x1, x2, tmp_sig));
    predictions = svmPredict(model, Xval);
    tmp_error = mean(double(predictions ~= yval));
    % fprintf('error = %f\n', tmp_error);
    % If the model scores higher than the current current_error
      % 1) Update the current_error to this values
      % 2) Update C; C := C_vec[i];
      % 3) Update sig; sig := Sig_vec[j];
    if tmp_error < current_error
      % fprintf('current_eror = %f\n', current_error);
      % fprintf('new_eror = %f\n', tmp_error);
      current_error = tmp_error;
      C = tmp_c;
      sigma = tmp_sig;
    endif
  endfor
endfor

% fprintf('\nFinal Values\ncurrent eror = %f\n', current_error);
% fprintf('C = %f\n', C);
% fprintf('sig = %f\n\n', sigma);
% =========================================================================

end
