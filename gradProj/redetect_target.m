function [occ, particle] = redetect_target(im,particle)
occ = true;
% check the median of the search result
target = get_subwindow(im,particle.gbest(1:2),particle.gbest(3:4));

% check if the median of the result, if is similar to pre_median, then it's
% the target
nodata = target==0;
newp = target(~nodata);
mmedian = median(double(newp));
if abs(mmedian-particle.pre_median)<800
    occ = false;
    particle.pre_median = mmedian;
    particle.pre_gbest = particle.gbest;
end

% figure(2);
[f,x] = ksdensity(double(newp),'Bandwidth',40);
% plot(x,f,'b-');

maxValue = max(f);
[density,posPeak] = findpeaks([0;f';0], 'MINPEAKDISTANCE',5,'MINPEAKHEIGHT',0.02*maxValue);

end