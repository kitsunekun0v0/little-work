function [hog_features, dhog_features] = get_hog_features(rgb_img, depth_img)

cell_size = 8;
norient = 9;
dhog_features = [];
nbins = 8;
nwindow = 6;

% extracting hog features from rgb and depth images
hog_features = double(fhog(single(rgb_img)/255, cell_size, norient));
dhog_features = double(fhog(single(depth_img)/255, cell_size, norient));

% remove all-zeros channel
hog_features(:,:,end) = [];
dhog_features(:,:,end) = [];

% pixel intensity histogram, from Piotr's Toolbox
h1=histcImWin(rgb_img,nbins,ones(nwindow,nwindow),'same');        
h1=h1(cell_size:cell_size:end,cell_size:cell_size:end,:);

% intensity adjusted hitorgram
% im= 255-calcIIF(rgb_img,[cell_size,cell_size],32);
% h2=histcImWin(im,nbins,ones(nwindow,nwindow),'same');
% h2=h2(cell_size:cell_size:end,cell_size:cell_size:end,:);

%concatenate depth and rgb hog features
%hog_features = cat(3,hog_features,dhog_features);
hog_features = cat(3,hog_features,dhog_features,h1);

end