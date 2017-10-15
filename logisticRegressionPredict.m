function prediction = logisticRegressionPredict(beta, testSet)

% INPUT: 
%     beta: coeffiecients associated with each attribute obtained from the logisticRegressionModelFcn.m
%     testSet: data from the BCWD table that were set aside to test the predictive power of our model 
% 
% OUTPUT:
%     prediction: list of predicted response variables for our test data
    
prediction = zeros(length(testSet), 1);
for j = 1:size(testSet, 2)
    % apply model coefficients to logistic regression equation to obtain a
    % value between 0 and 1.  A predicted value (p) less than 0.5 will be
    % classified as benign and values above 0.5 will be classified as
    % malignant.
    p = 1/(1+exp(-(beta(1) + sum(beta(2:end) .* testSet(j,:)))));
    if p < 0.5
        prediction(j) = 0;
    else
        prediction(j) = 1;
    end
end


