function label = k_means(C, X)

dC = Inf;
while dC>1
    pre_C = C;
    % find the distance between data point and cluster centers.
    c2_2xc = bsxfun(@minus,dot(C',C',1)',2*C*X');
    D = bsxfun(@plus,c2_2xc',dot(X',X',1)');
    D = D';
    [~,label] = min(D,[],1);

    % recalculate cluster centers.
    for i = 1:k
        C(i) = sum(X(label'==i))/size(X(label'==i),1);
    end
    dC=max(abs(C-pre_C));
end

end
