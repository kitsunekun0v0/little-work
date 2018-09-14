function [result,C] = target_segmentation(im,x,posPeak)
[h,w] = size(im);
% data point is n dimension array. n is the number of pixel. 
newp = double(im(:));
nodata = newp==0; % find missing data
newp(nodata) = median(newp);% assign missing data some value. use median.

C = x(posPeak); % initialise cluster centers.
k = length(C);
C = C';
dC = Inf;

while dC>1
    pre_C = C;
    % find the distance between data point and cluster centers.
    c2_2xc = bsxfun(@minus,dot(C',C',1)',2*C*newp');
    D = bsxfun(@plus,c2_2xc',dot(newp',newp',1)');
    D = D';
    [~,label] = min(D,[],1);
    % recalculate cluster centers.
    for i = 1:k
        C(i) = sum(newp(label'==i))/size(newp(label'==i),1);
    end
    dC=max(abs(C-pre_C));
end

label(nodata) = k+1;
result = reshape(label,[h,w]);
end