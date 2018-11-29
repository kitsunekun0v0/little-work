function [weight, learning_curve] = q_learning(n_episode,lr,eps,gamma)
% n_episode: number of episode.
% lr: learning rate.
% gamma: reward decay.
% eps: epsilon greedy.

%% define the environment
N = 7; M = 10; % # of row and column of the gridworld
n_state = N*M; % totle # f states
state_matrix = eye(n_state); % represent each state by a vector

n_action = 4; % # of action
v_change = [-1 0 1 0]; %vertical movement
h_change = [0 1 0 -1]; %horizontal movement

T = [4 8]; %terminal
end_state = sub2ind([N,M],T(1),T(2));

R = 10; %reward at the terminal state
max_step = 150; %maximum steps for each episode
weight = rand(n_action,n_state); %initialise the weight
learning_curve = zeros(1,n_episode);

%% Q Learning
for episode = 1:n_episode
    S = [randi(N) randi(M)]; %randomly set a start state
    start_state = sub2ind([N,M],S(1),S(2));
    
    current_state = S; %set current state
    index = start_state;
    step = 0;
    is_terminal = false;
    
    %for each step in an episode
    while ~is_terminal && (step<=max_step)
        step = step + 1;
        r = 0;
        input = state_matrix(:,index); %convert state into an input vector
        
        %compute Q value of current state and select an action. 
        %Q value = exp(weight*input)/sum(exp(weight*input))
        Q = exp(weight*input)/sum(exp(weight*input));
        %choose an action with epsilon greedy policy
        if rand>eps
            action = find(Q==max(Q));
        else
            action = randi(n_action);
        end
        q_predict = Q(action);
        
        action_vec = zeros(n_action,1);
        action_vec(action,1) = 1;
        
        %observe new state
        new_state(1) = current_state(1) + v_change(action);
        new_state(2) = current_state(2) + h_change(action);
        
        %not move out of the gridworld
        new_state(1) = (new_state(1)<1) + new_state(1)*((new_state(1)>0)&&(new_state(1)<=N)) + N*(new_state(1)>N);
        new_state(2) = (new_state(2)<1) + new_state(2)*((new_state(2)>0)&&(new_state(2)<=M)) + M*(new_state(2)>M);
        
        new_index = sub2ind([N,M],new_state(1),new_state(2));
        
         
        if new_index==end_state %check if state is terminal
            dw = (R-q_predict)*(action_vec-Q)*input';
            weight = weight + lr*dw;
            is_terminal = true;
            
        else
            next_input = state_matrix(:,new_index);%next state
            q_target = exp(weight*next_input)/sum(exp(weight*next_input));
            action = find(q_target==max(q_target));
            
            %update Q value
            dw = (r - q_predict + gamma*q_target(action))*(action_vec-Q)*input';
            weight = weight + lr*dw;
            
            index = new_index;
            current_state(1) = new_state(1);
            current_state(2) = new_state(2);
        end
    
    learning_curve(1, episode) = step;
end

end

end