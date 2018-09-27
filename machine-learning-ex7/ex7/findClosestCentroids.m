function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

# Tmp magnitudes/variances
tmp_idx = zeros(K, 1);

for i = 1:length(idx)
	x = X(i, :);

	% for j = 1:K
	% 	fprintf('size cent = [%d %d]\n', size(x));
	% 	tmp = x - centroids(j, :);
	% 	tmp = abs(tmp(2) - tmp(1));
	% 	tmp_idx(j) = tmp^2;
	% endfor

	tmp_idx = abs(x - centroids);
	tmp_idx = sum(tmp_idx.^2, 2).^0.5;
	tmp_idx = tmp_idx.^2;

	% Get the index - aka. idx - for the centroid that minimized "abs(x - centroids(j))^2"
	% fprintf('min idx(%d) = [%d %d]\n', [i; find(tmp_idx == min(tmp_idx))]);
	idx(i) = find(tmp_idx == min(tmp_idx))(1);

endfor



% =============================================================

end
