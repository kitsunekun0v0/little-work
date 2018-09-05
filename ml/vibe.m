function mask = vibe(frame)
clc
%load frames450

% width and height of the video
rows = 336;
cols = 448;

%size of sample = 20;
%minMatch = 2;
%updateFactor = 16;
%disThres = 20;

% neighbour pixels
nx = [-1,0,1,1,-1,-1,1,0,0];
ny = [-1,0,1,-1,1,0,0,1,-1];
% background model
bgModel = zeros(rows,cols,3,20,'uint8');
b_mask = zeros(rows,cols);
fg_count = zeros(rows,cols);

%% initialise background model
% frame = a(:,:,:,1);
% for r=2:rows-1
%     for c=2:cols-1
%         for s=1:20
%             ran_i = randi(9);
%             rr = r+ny(ran_i);
%             cc = c+nx(ran_i);
%             
%             bgModel(r,c,1,s) = frame(rr,cc,1);
%             bgModel(r,c,2,s) = frame(rr,cc,2);
%             bgModel(r,c,3,s) = frame(rr,cc,3);
%         end
%     end
% end

%% test new frame and update model
for i=2:150
    for r=2:rows-1
        for c=2:cols-1
            count=0;
            
            for s=1:20
                % calculate distance between pixel and sample
                d = sqrt((double(bgModel(r,c,1,s))-double(frame(r,c,1)))^2 + ...
                    (double(bgModel(r,c,2,s))-double(frame(r,c,2)))^2 + ...
                    (double(bgModel(r,c,3,s))-double(frame(r,c,3)))^2);
                
                if d<21 % if the difference is small
                    count = count+1;
                end
                
                if count>2 % classify this pixel as background
                    b_mask(r,c) = 0;
                    fg_count(r,c) = 0;
                    
                    % probably update background model
                    ran_i = randi(10);
                    if ran_i==1
                        ran_i=randi(20);
                        bgModel(r,c,1,ran_i) = a(r,c,1,i);
                        bgModel(r,c,2,ran_i) = a(r,c,2,i);
                        bgModel(r,c,3,ran_i) = a(r,c,3,i);
                    end
                    
                    % probably update neighbour's background model
                    ran_i = randi(10);
                    if ran_i==1
                        ran_i = randi(9);
                        rr = r+ny(ran_i);
                        cc = c+nx(ran_i);
                        ran_i = randi(20);
                        bgModel(rr,cc,1,ran_i) = frame(r,c,1);
                        bgModel(rr,cc,2,ran_i) = frame(r,c,2);
                        bgModel(rr,cc,3,ran_i) = frame(r,c,3);
                    end
                    
                    % break the for loop
                    break
                    
                end 
            end
            
            if count<3
                fg_count(r,c) = fg_count(r,c)+1;
                b_mask(r,c) = 1;
                
                % if a pixel is always foreground, change it to background
                if fg_count(r,c)>10
                    ran_i = randi(9);
                    if ran_i==1
                        ran_i=randi(20);
                        bgModel(r,c,1,ran_i) = frame(r,c,1);
                        bgModel(r,c,2,ran_i) = frame(r,c,2);
                        bgModel(r,c,3,ran_i) = frames(r,c,3);
                    end
                end
            end
        end
    end
    
    %foremm = bwmorph(b_mask,'erode',1); % binary mask
    %foremm = bwmorph(b_mask,'dilate',1);
    %foremm(1:27,1:259) = 0; % ignore top left corner
    %s = regionprops(b_mask,'basic'); %get info about detected region
    
    %[N,W] = size(s);
    %if N~=0
%       figure(1)
%       imshow(a(:,:,:,i))  
%       hold on
%       for ii=1:N
%           centroid=s(ii).Centroid;
%           plot(centroid(:,1),centroid(:,2),'r*')
%       end
%       hold off
%     end

Iblur1 = imgaussfilt(b_mask,1);
Iblur1 = Iblur1>0.7;

%     figure(2)
%     imshow(Iblur1)
   mask = Iblur1; 
end
end






