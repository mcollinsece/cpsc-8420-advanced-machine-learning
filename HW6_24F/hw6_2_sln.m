clear all
close all
clc

% Homework 6.2

datasets = {'text.mat', 'concentric.mat', 'links.mat', 'rectangles.mat'};
data_vars = {'X4', 'X1', 'X3', 'X2'};
dataset_names = {'text', 'concentric', 'links', 'rectangles'}; 
sigma_values = [0.025, 0.05, 0.2, 0.5];

for i = 1:length(datasets)
    load(datasets{i});
    var_name = data_vars{i};
    if exist(var_name, 'var')
        data = eval(var_name); 
    else
        error(['Variable ', var_name, ' not found in dataset ', datasets{i}]);
    end

    if strcmp(datasets{i}, 'text.mat')
        k = 6; 
    else
        k = 2; 
    end

    for j = 1:length(sigma_values)
        sigma = sigma_values(j);
        idx = spectralClustering(data, k, sigma);
   
        plotClusters(data, idx);
        title(['Spectral Clustering for ', datasets{i}, ', \sigma = ', num2str(sigma)]);
        
        filename = sprintf('%s_sigma_%.3f.png', dataset_names{i}, sigma);
        saveas(gcf, filename);
    end
end

function [idx] = spectralClustering(data, k, sigma)
    n = size(data, 1);
    A = zeros(n, n);
    for i = 1:n
        for j = 1:n
            A(i, j) = exp(-norm(data(i, :) - data(j, :))^2 / sigma);
        end
    end

    D = diag(1 ./ sqrt(sum(A, 2)));
    N = D * A * D; 

    [V, ~] = eigs(N, k);
    Y = V ./ vecnorm(V, 2, 2);
    [idx, ~] = kmeans(Y, k);
end
