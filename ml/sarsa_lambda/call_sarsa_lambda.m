n_episode = 50;
lr = 0.9; %between 0 and 1
epsilon = 0.3; %<=0. epsilon=0 greedy, otherwise epsilon-greedy
gamma = 0.9; %reward decay. positive, smaller than 1.
lambda = 0.8;

n_repetition = 100;
learning_curve = zeros(n_repetition,n_episode);
for i = 1:n_repetition
    [w,s] = sarsa_lambda(n_episode,lr,epsilon, gamma, lambda);
    learning_curve(i,:) = s;
end

%plot the average steps as a number of episodes.
h = errorbar(mean(learning_curve),2*std(learning_curve)./sqrt(n_repetition));
set(h,'linewidth',2);
xlabel('episodes'); ylabel('average steps');
set(gca, 'fontsize', 18);