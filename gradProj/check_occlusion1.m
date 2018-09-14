function [occ, p_occ, particle] = check_occlusion1(im, particle)
occ = false;
p_occ = false;
im = get_subwindow(im,particle.gbest(1:2),particle.gbest(3:4));
nodata = im==0;
newp = im(~nodata);
im_median = median(double(newp));

[h,w] = size(im);
hh = floor(h/2);
ww = floor(w/2);

% partitioning patch into 4 parts.
sub_1 = im(1:hh,1:ww);
sub_2 = im(hh+1:end,1:ww);
sub_3 = im(1:hh,ww+1:end);
sub_4 = im(hh+1:end,ww+1:end);

% check occlusion for each part. 
[occ_1,median_1] = check_subregion(sub_1);
[occ_2,median_2] = check_subregion(sub_2);
[occ_3,median_3] = check_subregion(sub_3);
[occ_4,median_4] = check_subregion(sub_4);

% if all parts are occluded, then occlusion.
if (occ_1 && occ_2 && occ_3 && occ_4)
    occ = true;
else 
    % update median depth if more than half of target isn't occluded
    if sum(double([occ_1,occ_2,occ_3,occ_4]))<2
        particle.pre_median = im_median;
    end

    % if partial occlusion appears, adjust the bounding box size by
    % calculating area that is not hidden by occluder. This is done by
    % segmenting the target region.
    
%      newp(double(newp)>9600) = 9600; % for face_occ5
%     newp(double(newp)<5500) = 5500;

    % kernel density estimation for finding the pdf of depth value
    [f,x] = ksdensity(double(newp),'Bandwidth',40); % bear:40
%     figure(6);
%     plot(x,f,'b-');

    % get peaks from kde
    maxValue = max(f);
    [density,posPeak] = findpeaks([0;f';0], 'MINPEAKDISTANCE',5);%,'MINPEAKHEIGHT',0.02*maxValue);
    posPeak(end) = posPeak(end)-1;
    
    % remove very close centers
    [~,depth_diff] = min(abs(x(posPeak)-particle.pre_median));
    remove_centers = abs(x(posPeak)-x(posPeak(depth_diff)));
    remove_centers = (remove_centers>0 & remove_centers<400); % child:300, bear:400, face:none
    posPeak = posPeak(~remove_centers);
    
    % find cluster that is closest to median
    depth_diff = abs(x(posPeak)-particle.pre_median);
    [~,diffIndex] = min(depth_diff);
    
    % if occluder appears, segment target
    if (diffIndex~=1) && (density(1)>density(diffIndex)/2)
%         disp('partial occ')
         p_occ = true;
%         % use k-means to segment the targets. return pixel label and
%         % cluster centers.
        [L,centers] = target_segmentation(im,x,posPeak);
% %         figure(2)
% %         imshow(label2rgb(L))
%         % binary mask of target object.
        mask = L==diffIndex;

%         % find the size and center of the target region
        bbox=regionprops('table',mask, 'BoundingBox');
% %         figure(3);
% %         imshow(mask);
%         
        b = bbox.BoundingBox;
        % target = the biggest area
        A = b(:,4).*b(:,3);
        [~,biggestIndex] = max(A);
        seg_sz = [b(biggestIndex,4),b(biggestIndex,3)];
        % find the location of bounding box center according to patch's top
        % left location.
        patch_topleft = particle.gbest(1:2)-floor(particle.gbest(3:4)/2);
        seg_pos = [b(biggestIndex,2)+patch_topleft(1),b(biggestIndex,1)+patch_topleft(2)]+floor(seg_sz/2);
        particle.partial_target = [seg_pos,seg_sz];
%         
     end
end

%% supporting function
    function [subocc,mmedian] = check_subregion(subr)
        subocc = false;
        nodata = subr==0;
        newp = subr(~nodata);
        mmedian = median(double(newp));
        if abs(mmedian-particle.pre_median)>600 %1000 %600
            subocc=true;
        end
    end

end