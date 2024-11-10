function plotClusters(dataset,clusteridx)

% PLOTCLUSTERS
% 
% function plotClusters(dataset,clusteridx)
% 
% dataset is an Nx2 or Nx3 matrix
% clusteridx is an N dimensional vector of cluster assignments
%   so clusteridx=[1 1 2 2] means that the first two points
%   belong to cluster one and the second two points belong to 
%   cluster two

marks = '.xo+*sdvph';
figure; 
for i=1:length(marks)
    C = find(clusteridx == i);
    if length(C)>0
        if size(dataset,2)==2
            plot(dataset(C,1),dataset(C,2),marks(i));
            hold on;
            axis equal
        else
            plot3(dataset(C,1),dataset(C,2),dataset(C,3), marks(i));
            hold on; axis equal; grid on;
        end
    end
end