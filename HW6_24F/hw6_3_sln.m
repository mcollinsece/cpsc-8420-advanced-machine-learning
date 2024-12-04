clear all
close all
clc

% Homework 6.3

datasets = {'rectangles.mat', 'text.mat'};
data_vars = {'X2', 'X4'}; 
sigma = 0.05; 

for i = 1:length(datasets)
    load(datasets{i});
    var_name = data_vars{i};
    if exist(var_name, 'var')
        data = eval(var_name); 
    else
        error(['Variable ', var_name, ' not found in dataset ', datasets{i}]);
    end

    n = size(data, 1);
    A = zeros(n, n);
    for p = 1:n
        for q = 1:n
            A(p, q) = exp(-norm(data(p, :) - data(q, :))^2 / sigma);
        end
    end

    D = diag(1 ./ sqrt(sum(A, 2)));
    N = D * A * D;

    [V, E] = eig(N); 
    eigenvalues = diag(E); 
    eigenvalues = sort(eigenvalues, 'descend');

    figure;
    plot(1:10, eigenvalues(1:10), '-o');
    title(['Top 10 Eigenvalues for ', datasets{i}, ' (\sigma = 0.05)']);
    xlabel('Index');
    ylabel('Eigenvalue');
    filename = sprintf('first_10_eigenvalues_%s.png', datasets{i}); 
    saveas(gcf, filename);
end
