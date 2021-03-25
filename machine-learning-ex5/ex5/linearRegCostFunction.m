function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

mInverse = 1 / m;
sumOfElementSq = @(x) sum(sum(x .* x));
thetaWithoutFirstElement = theta(2 : end);

cost = X * theta - y;

J = 0.5 * mInverse * (sumOfElementSq(cost) + lambda * sumOfElementSq(thetaWithoutFirstElement));

grad = mInverse * (X' * cost + lambda * [0; thetaWithoutFirstElement]);

% =========================================================================

% grad = grad(:);

end
