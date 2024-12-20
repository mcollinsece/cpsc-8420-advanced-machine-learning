clear all
close all
clc

% Homework 6.1

datasets = {'text.mat', 'concentric.mat', 'links.mat', 'rectangles.mat'};
data_vars = {'X4', 'X1', 'X3', 'X2'}; % Variable names in the .mat files

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

    [idx, ~] = kmeans(data, k);

    plotClusters(data, idx);
    title(['K-means Clustering for ', datasets{i}, ', k = ', num2str(k)]);
    filename = sprintf('k_means_clustering_%s.png', datasets{i}); 
    saveas(gcf, filename);
end
