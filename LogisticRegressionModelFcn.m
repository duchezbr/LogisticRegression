function [beta, accuracy] = LogisticRegressionModelFcn(X, y, alpha, epochs)

% PURPOSE;
%       
% INPUT:
%       X: MxN matrix where rows represent observations for N number
%       of attributes
%       y: Mx1 list of binary classification variables
%       alpha: offest that will be used to adjust the beta
%       coefficients for eaching training run
%       epochs: number of iterations through the training data set, X.
%
% OUTPUT:
%       beta: are the coefficients for the training model 
%       accuracy: is a Mx1 list of the percentage of correctly identified
%       classification labels following each epoch (iteration through all
%       rows of the training data, X).

rows = size(X, 1);
coefficients = size(X, 2) + 1;
beta = zeros(1, coefficients);
accuracy = [];

for i = 1:epochs
    
    score = zeros(rows, 1);

    for j = 1:rows

        prediction = 1/(1+exp(-(beta(1) + sum(beta(2:end) .* X(j,:)))));


        % update coeffiecients
        beta(1) = beta(1) + alpha*(y(j) - prediction)*prediction*(1-prediction)*1;

        for k = 2:coefficients
     
            beta(k) = beta(k) + alpha*(y(j) - prediction)*prediction*(1-prediction).*X(j, k-1);

        end

        % I can let the user decide where to set the cut-off
        if prediction < 0.5
            score(j) = 0;
        else
            score(j) = 1;
        end
    end
    % tabulate the percentage of correctly classified labels from this
    % iteration through the data
    correctlyIdentified = sum(all(score == y,2));
    accuracy = [accuracy; correctlyIdentified/length(y)];

end
figure;
plot(accuracy)
xlabel('Epoch')
ylabel('Percent')