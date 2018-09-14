function [rgb_fn, depth_fn] = load_filename(pa)
load(pa);
rgb_fn = {};
depth_fn = {};
noF = frames.length;
for i = 1:noF
    rgb_fn{i} = sprintf('r-%d-%d.png', frames.imageTimestamp(i),frames.imageFrameID(i));
    depth_fn{i} = sprintf('d-%d-%d.png', frames.depthTimestamp(i),frames.depthFrameID(i));
end
end