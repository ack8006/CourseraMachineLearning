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


%%inner = sum(((X*theta) - y).^2);
%inner = ((X*theta) - y);
%reg = (lambda/(2*m))*(sum(theta(2:end).^2));
%
%J = (1/(2*m)) * sum(inner.^2) + reg;
%
%%size(X) 12x2
%%size(y) 12x1
%%size(theta) 2x1
%%size(inner) 12x1
%
%grad = (1/m)*sum((inner).*X);
%reg = [zeros(size(theta(1:1))) (lambda/m*theta(2:end))];
%grad = grad+reg;

regTheta = theta(2:end);
Herr = (X * theta) - y;
nonreg = (1/(2*m)) * sum(Herr.^2);
reg = (lambda / (2*m)) * (regTheta' * regTheta);

J = nonreg + reg;


grad = (X' * Herr) / m;
grad(2:end) += lambda * regTheta/m;

%size(theta)
%size(J)
%size(grad)
%2x1, 1x1, 2x1




% =========================================================================

grad = grad(:);

end
