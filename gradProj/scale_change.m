function [particle,r,pre_h_] = scale_change(depth_im, particle, init_median, init_size,pre_h)
    % r = (previous frame median)/(current frame median), rescale ration
%     target_roi_depth = get_subwindow(depth_im,particle.gbest(1:2),particle.gbest(3:4));
%     nodata = target_roi_depth==0;
%     target_roi_depth = target_roi_depth(~nodata);
%     mmedian = median(double(target_roi_depth(:)));
    r = init_median/particle.pre_median;
   
    % update gbest
    particle.gbest(3:4) = init_size*r;
    
    % update weight and height of target
    particle.w = particle.gbest(4);
    particle.h = particle.gbest(3);
%     particle.pre_median = mmedian;
    pre_h_ = pre_h;
    [particle,pre_h_] = shape_change(depth_im,particle,pre_h);

end