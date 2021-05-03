function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

Cs = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmas = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

sizeOfCs = size(Cs, 1);
sizeOfSigmas = size(sigmas, 1);

errors = zeros(sizeOfSigmas);

i = 1;
minError = 1;
minErrorIdx = 1;
bestC = Cs(1);
bestSigma = sigmas(1);
for j = [1:sizeOfCs]
  C_try = Cs(j);
  for k = [1:sizeOfSigmas]
    sigma_try = sigmas(k);
    model_try = svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try)); 
    predictions_try = svmPredict(model_try, Xval);
    error = mean(double(predictions_try ~= yval));

    if (error < minError)
      minError = error;
      minErrorIdx = i;
      bestC = C_try;
      bestSigma = sigma_try;
    end

    i
    C_try
    sigma_try
    error

    errors(i) = error;
    i++;
  end
end

C = bestC;
sigma = bestSigma;

% =========================================================================

end
