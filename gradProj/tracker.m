function positions = tracker()
addpath('feature');
addpath('New Folder');

%% load files information
%base_path = 'ValidationSet/new_ex_occ4/';
%base_path = 'ValidationSet/child_no1/';
base_path = 'ValidationSet/zcup_move_1/';
%base_path = 'ValidationSet/bear_front/';
%base_path = 'ValidationSet/face_occ5/';

f_name = [base_path 'init.txt']; %ground truth file
path_depth = [base_path 'depth/'];
path_rgb = [base_path 'rgb/'];

% list all depth maps and images
rgb_im_ = dir([path_rgb '*.png']);
assert(~isempty(rgb_im_), 'no image loaded')
depth_im_ = dir([path_depth '*.png']);
assert(~isempty(depth_im_), 'no depth map loaded')

pa = [base_path 'frames'];
[rgb_fn, depth_fn] = load_filename(pa);

% load initial target data
f = importdata(f_name);
init_sz = [f(1,4),f(1,3)]; % get size of the target (h,w)
init_pos = [f(1,2),f(1,1)]+floor(init_sz/2); % get initial target position (y,x). move from top left to center.

positions = zeros(numel(rgb_fn),4); % to store position. [x,y,w,h]
iteration = 10; % no of iteration
occ = false; 
p_occ = false;

%% processing each frame
for f = 1:numel(rgb_fn)
    X = sprintf('%d frame',f);
    disp(X);
    % load image and depth map
    rgb_im = imread([path_rgb rgb_fn{f}]);
    depth_im = imread([path_depth depth_fn{f}]);
    
    % if it's the first frame, initialise particles and store initial 
    % target depth.
    if f==1
        % extract patches contain target object
        target_rgb = get_subwindow(rgb_im, init_pos, init_sz);
        target_depth = get_subwindow(depth_im, init_pos, init_sz);
        
        % get combination of rgb and depth hog features. 
        rnd_hog = get_hog_features(target_rgb, target_depth);
        [x,y,z] = size(rnd_hog);
        rnd_hog = reshape(rnd_hog, [1, x*y*z]);
        rnd_hog = rnd_hog/(max(rnd_hog)-min(rnd_hog));
        
        % find the median depth value from the patch
        nodata = target_depth==0;
        target_depth = target_depth(~nodata);
        init_median = median(double(target_depth(:)));
        
        % initialising particle structure
        particle = setup_particle();
        pre_hw = [particle.h;particle.w];
    end
    
    
    % when no occlusion in previous frame
    if ~particle.pre_occ
        init_particle(); % place particles randomly

        % iteratively update particles
        for i = 1:iteration
            % calculate the fitness and update the particles
            update_particle();
            if i==iteration
                break;
            end
            move_swarm();
        end


        % check occlusion
        if f~=68
         [occ,p_occ,particle] = check_occlusion1(depth_im, particle);
        else
            occ = true;
        end
        
        if ~occ % no occlusion detect in this frame, normally track object
           
           % if object is partially occluded, size and position of bounding
           % box depend on target segmentation result. 
           if ~p_occ
               % change scale
               [particle,~,pre_hw] = scale_change(depth_im, particle, init_median, init_sz,pre_hw);
               positions(f,:) = [particle.gbest(2), particle.gbest(1), particle.gbest(4), particle.gbest(3)];
           else
               positions(f,:) = [particle.partial_target(2),particle.partial_target(1),particle.partial_target(4),particle.partial_target(3)];
           end
            disp(particle.gbest_score);

            % update pre_gbest and reset gbest_score
            particle.pre_gbest = particle.gbest;
            particle.gbest_score = 0; 
            
        else % there's an occlusion in this frame
            % do not record the predicted position
            positions(f,:) = [NaN, NaN, NaN, NaN];
            % no need to update median depth and particles state
            % wider the search space
            particle.bound = 100;
            particle.n_particles = 200;
            particle.particles = zeros(9, particle.n_particles);
            particle.pre_gbest = particle.gbest;
            disp('occlusion');
            disp(particle.gbest_score);
            particle.gbest_score = 0;
        end
    end
    
    
    % when there is occlusion in previous frame
    if particle.pre_occ
        % search target
        init_particle();
        for i = 1:iteration
            update_particle();
            if i==iteration
                break;
            end
            move_swarm();
        end
        
        % check if PSO redetects the target
        [occ, particle] = redetect_target(depth_im, particle); 
        
        % if target appears
        if ~occ
            particle.bound = 20;
            particle.n_particles = 30;
            particle.particles = zeros(9,particle.n_particles);
            positions(f,:) = [particle.gbest(2),particle.gbest(1), particle.gbest(4), particle.gbest(3)];
        else
            positions(f,:) = [NaN, NaN, NaN, NaN];
        end
        
        disp(particle.gbest_score);
        particle.gbest_score = 0;
    end
    
    particle.pre_occ = occ;
    
    % ++++++++++++++++++++++visualisation part++++++++++++++++++++++++++
    % displace gbest and save result
    if ~occ
        rect = positions(f,:);
        rect = [rect(1)-rect(3)/2.0,rect(2)-rect(4)/2.0,rect(3),rect(4)];
    else
        rect = particle.gbest;
        rect = [rect(2)-rect(4)/2.0, rect(1)-rect(3)/2.0, rect(4), rect(3)];
    end
    
    figure(1);
    imshow(rgb_im, 'border', 'tight');   
    hold on
    % show gbest when occlusion appears
        rectangle('Position', rect, 'Linewidth', 2, 'edgecolor', 'red');
    % indicate occlusion if it happened
    if particle.pre_occ
        text(25,25,'occlusion','color','r','fontsize',20,'fontweight','bold');
    end
    hold off

    frm = getframe(gca);
    imwrite(frm.cdata, strcat('result/', sprintf('%4.4d',f),'.png'))
end



%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% supporting function
%  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% setup particle attributes
    function particle = setup_particle()
        particle.n_particles = 30; % number of particle
        particle.particles = zeros(9,particle.n_particles); % particles. 
        %column = each particles. row = attribute. 1:2=coordinate (y,x), 
        % 3:4=height and weight, 5:6=velocity, 7:8 = coordinate of pbest.
        % 9=pbest.
        particle.w = init_sz(1,2); % width of target
        particle.h = init_sz(1,1); % height of target
        particle.start_pos = init_pos;
        particle.gbest = [init_pos,init_sz].';
        particle.pre_gbest = [init_pos,init_sz].';
        particle.gbest_score = 0;
        particle.bound = 20;
        particle.partial_target = zeros(1,4);
        particle.pre_occ = false;
        particle.pre_median = init_median;
    end

%% initialise particles
    function init_particle()
       % randomly put the particles in search space
       particle.particles(1,:) = rand(1,particle.n_particles)*particle.bound*2+particle.pre_gbest(1)-particle.bound;
       particle.particles(2,:) = rand(1,particle.n_particles)*particle.bound*2+particle.pre_gbest(2)-particle.bound;
       particle.particles(3,:) = ones(1,particle.n_particles)*particle.h;
       particle.particles(4,:) = ones(1,particle.n_particles)*particle.w;
    end

%% find the fitness
    function update_particle()
        for p = 1:particle.n_particles
            particle_r_patch = get_subwindow(rgb_im,particle.particles(1:2,p),particle.particles(3:4,p));
            particle_d_patch = get_subwindow(depth_im,particle.particles(1:2,p),particle.particles(3:4,p));
            particle_r_patch = imresize(particle_r_patch, init_sz, 'bilinear');
            particle_d_patch = imresize(particle_d_patch, init_sz, 'bilinear');
            
            % extract hog features from patch
            rnd_hog_p = get_hog_features(particle_r_patch, particle_d_patch);
            [x_,y_,z_] = size(rnd_hog_p);
            rnd_hog_p = reshape(rnd_hog_p, [1, x_*y_*z_]);
            rnd_hog_p = rnd_hog_p/(max(rnd_hog_p)-min(rnd_hog_p));
            fitness = dot(rnd_hog_p,rnd_hog)/(norm(rnd_hog,2)*norm(rnd_hog_p,2));
            
            % update gbest and pbest according to finess
            % update pbest:
            if fitness > particle.particles(9,p)
                particle.particles(9,p) = fitness;
                particle.particles(7:8) = particle.particles(1:2,p);
            end
            % update gbest:
            if fitness > particle.gbest_score
                particle.gbest_score = fitness;
                particle.gbest = particle.particles(1:4,p);
            end
        end    
    end


%% replace the particles according to velocity
    function move_swarm()
        % v = wv+c1r1(pbest-x)+c2r2(gbest-x)
        w = 1.0;
        c1 = 2.0;
        c2 = 2.0;
        r1 = rand([2,particle.n_particles]);
        r2 = rand([2,particle.n_particles]);
        
        particle.particles(5:6,:) = w*particle.particles(5:6,:) + c1*r1.*(bsxfun(@minus,particle.particles(1:2,:),particle.gbest(1:2))) + ...
            c2*r2.*(particle.particles(7:8,:)-particle.particles(1:2,:));
        
        % limite the velocity
        temp = particle.particles(5,:);
        temp(temp<-200) = -200;
        particle.particles(5,:) = temp;
        
        temp = particle.particles(5,:);
        temp(temp>200) = 200;
        particle.particles(5,:) = temp;
        
        temp = particle.particles(6,:);
        temp(temp<-200) = -200;
        particle.particles(6,:) = temp;
        
        temp = particle.particles(6,:);
        temp(temp>200) = 200;
        particle.particles(6,:) = temp;
        
        % update location
        particle.particles(1:2,:) = particle.particles(5:6,:) + particle.particles(1:2,:);
        
        % particles have to be within seach space
        temp = particle.particles(1,:);
        temp(temp<particle.pre_gbest(1)-particle.bound) = particle.pre_gbest(1)-particle.bound;
        particle.particles(1,:) = temp;
        
        temp = particle.particles(1,:);
        temp(temp>particle.pre_gbest(1)+particle.bound) = particle.pre_gbest(1)+particle.bound;
        particle.particles(1,:) = temp;
        
        temp = particle.particles(2,:);
        temp(temp<particle.pre_gbest(2)-particle.bound) = particle.pre_gbest(2)-particle.bound;
        particle.particles(2,:) = temp;
        
        temp = particle.particles(2,:);
        temp(temp>particle.pre_gbest(2)+particle.bound) = particle.pre_gbest(2)+particle.bound;
        particle.particles(2,:) = temp;
        
    end

end