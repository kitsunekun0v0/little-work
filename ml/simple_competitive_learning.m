function simple_competitive_learning(X,l,epoch,nn)
%X: nxD matrix. n samples, D features.
%l: learning rate.
%epoch: number of iteration.
%nn: number of output neurons.

[n,m] = size(X);

%initialise weight matrix W (n_neurons x n_features)
W = rand(nn,m);
%normalise weight as well as input vectors for convinience. 
normW = sqrt(diag(W*W'));
W = W./repmat(normW,1,m);
normX = sqrt(diag(X*X'));
X = X./repmat(normx,1,m);

%count how many time a neuron is a winner
counter = zeros(nn,1);

for t=1:epoch
    %randomly choose one sample from dataset and calculate activity for
    %each neuron.
    i = ceil(n*rand);
    x = X(i,:); %1xm
    h = W*x'; % activity of each neuron
    
    % add noise on activity to reduce number of dead neurons. Select the
    % winner of this iteration and update its weight.
    ns = rand(nn,1)/200;
    [~,winner] = max(h + ns);
    W(winner,:) = W(winner,:) + l*(x-W(winner,:));
    
    counter(winner,1) = counter(winner,1) + 1;
end

%dead neurons
dead_n = counter(counter<(epoch*0.2));

end