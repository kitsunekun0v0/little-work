function [particle,pre_hw] = shape_change(im,particle,pre_hw)
% reset size and position to enlarge bounding box horizontally and vertically.
% convert the depth maps into grayscale image, then do the segmentation.


%% verticle part
% segmentation for verticle part.
new_image = get_subwindow(im, particle.gbest(1:2),particle.gbest(3:4));
vp = im2uint8(mat2gray(double(new_image))); % convert to grayscale image
nodata = vp==0;
vp1 = vp(~nodata);

% find peaks
[f,x] = ksdensity(double(vp1(:)),'Bandwidth',1.0);
[density,posPeak] = findpeaks([0;f';0], 'MINPEAKDISTANCE',5);
posPeak(end) = posPeak(end)-1;

% segmentation
[~,peakIndex] = max(density); % target is the one with highest peak
L = target_segmentation(vp,x,posPeak);
mask1 = L==peakIndex;

% find the size and center of the target region
bbox=regionprops('table',mask1, 'BoundingBox');
b = bbox.BoundingBox;
A = b(:,4).*b(:,3);
[~,biggestIndex] = max(A);

% a sudden large change means segmentation error, ignore this segmentation
if abs(b(biggestIndex,4)-pre_hw(1))<29
    h = b(biggestIndex,4)+3;
    % relocate the bounding box center based on segmentation
    particle.gbest(1) = (b(biggestIndex,2)+particle.gbest(1)- ...
        floor(particle.gbest(3)/2)-3)+floor(h/2);
    particle.gbest(3) = h;
    pre_hw(1) = b(biggestIndex,4);
else
    particle.gbest(3) = pre_hw(1)+3;
end


%% horizontal part
% extract patch with new size
pos = particle.gbest(1:2);
sz = [particle.gbest(3)*0.7;particle.gbest(4)*1.1];
patch = get_subwindow(im,pos,sz);
new_image = im2uint8(mat2gray(double(patch))); % convert to grayscale image
nodata = new_image==0;
new_image1 = new_image(~nodata);

% find peaks
[f,x] = ksdensity(double(new_image1(:)),'Bandwidth',1.0);
% figure(2);
% plot(x,f,'b-');
[density,posPeak] = findpeaks([0;f';0], 'MINPEAKDISTANCE',5);
posPeak(end) = posPeak(end)-1;

% segmentation
[~,peakIndex] = max(density); % target is the one with highest peak
L = target_segmentation(new_image,x,posPeak); % return label map
mask = L==peakIndex; % binary map for target

% find the size and center of the target region
bbox=regionprops('table',mask, 'BoundingBox');
b = bbox.BoundingBox;
A = b(:,4).*b(:,3);
[~,biggestIndex] = max(A);

% change the width of the bounding box based on segmentation
% if abs(b(biggestIndex,3)-pre_hw(2))<13
    particle.gbest(4) = b(biggestIndex,3); 
    % relocate the bounding box center based on segmentation
    particle.gbest(2) = (b(biggestIndex,1)+pos(2)-floor(sz(2)/2))+floor(particle.gbest(4)/2);
    particle.gbest(4) = particle.gbest(4)+6; % enlarge margin
    pre_hw(2) = b(biggestIndex,3);
else
    particle.gbest(4) = pre_hw(2)+6;
end


% figure(3);
% subplot(1,2,1);
% imshow(mask);
% subplot(1,2,2);
% imshow(mask1);

end

