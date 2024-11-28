clear; clc; close all;

load usps_digital.mat;

learning_rate = 1e-4;   
max_iterations = 5000;  
lambdas = [0, 0.1, 1, 10, 100, 200]; 

final_test_accuracies = zeros(size(lambdas));
final_train_accuracies = zeros(size(lambdas));

for idx = 1:length(lambdas)
    lambda = lambdas(idx); 
    
    [B, test_error, train_error, objective_values] = log_reg(tr_y, tr_X, te_y, te_X, lambda, learning_rate);
    
    final_test_accuracies(idx) = test_error(end);
    final_train_accuracies(idx) = train_error(end);
    
    figure;
    plot(1:length(objective_values)-1, objective_values(1:end-1), 'b-o', 'LineWidth', 1, 'MarkerSize', 1);
    title(sprintf('Objective Function (\\lambda = %.1f)', lambda), 'FontSize', 15);
    xlabel('Number of Iterations', 'FontSize', 15);
    ylabel('Objective Value', 'FontSize', 15);
    set(gca, 'FontSize', 12);
    saveas(gcf, sprintf('objective_lambda_%.1f.fig', lambda));
    
    figure;
    plot(1:length(train_error)-1, train_error(1:end-1), 'r-o', 'LineWidth', 1, 'MarkerSize', 1);
    title(sprintf('Training Accuracy (\\lambda = %.1f)', lambda), 'FontSize', 15);
    xlabel('Number of Iterations', 'FontSize', 15);
    ylabel('Training Accuracy', 'FontSize', 15);
    set(gca, 'FontSize', 12);
    saveas(gcf, sprintf('train_accuracy_lambda_%.1f.fig', lambda));
    
    figure;
    plot(1:length(test_error)-1, test_error(1:end-1), 'g-o', 'LineWidth', 1, 'MarkerSize', 1);
    title(sprintf('Testing Accuracy (\\lambda = %.1f)', lambda), 'FontSize', 15);
    xlabel('Number of Iterations', 'FontSize', 15);
    ylabel('Testing Accuracy', 'FontSize', 15);
    set(gca, 'FontSize', 12);
    saveas(gcf, sprintf('test_accuracy_lambda_%.1f.fig', lambda));
end

fprintf('Final Testing and Training Accuracies for Each \\lambda:\n');
fprintf('-------------------------------------------------------\n');
fprintf('Lambda\tFinal Train Accuracy\tFinal Test Accuracy\n');
fprintf('-------------------------------------------------------\n');
for idx = 1:length(lambdas)
    fprintf('%.1f\t\t%.4f\t\t\t%.4f\n', lambdas(idx), final_train_accuracies(idx), final_test_accuracies(idx));
end
fprintf('-------------------------------------------------------\n');

save('final_accuracies.mat', 'lambdas', 'final_train_accuracies', 'final_test_accuracies');