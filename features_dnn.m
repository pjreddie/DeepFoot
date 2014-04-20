function feat = features_dnn(imds)
% function that takes an image name and loads its DNN features at the base
% resolution

try
    
pathToFeatures = '/projects/grail/unikitty/VOC2007/imagenet_features/4/';   % hardcoded path
interval = 4;                                                          % hardcoded interval

% imds has 'im', 'flip', 'boxes' (if positive); see pascal_data.m
alltoks = strtokAll(imds.im, '/');
fname = [pathToFeatures '/' alltoks{end} '.txt'];
% flip features if needed
if imds.flip
    fname = [pathToFeatures '/' alltoks{end} '_r.txt'];
end


fid = fopen(fname, 'r');

fcnt = 1;
while fcnt ~= interval+1    % ignore first "interval" lines as they all correspond to 2X resolution
    fgetl(fid); % ignore header
    fgetl(fid); % ignore data
    fcnt = fcnt + 1;
end

A = textscan(fid, '%f %f %f\n', 1, 'delimiter', ',');
ndim = A{1}; height = A{2}; width = A{3};
feat_tmp = textscan(fid, '%f', ndim*height*width, 'delimiter', ',');
fscanf(fid, '\n');
%feat = fscanf(fid, '%f', [ndim*height*width]);

feat = zeros(height, width, ndim+1, 'single');  % +1 is for truncation dim
cnt = 1;
for k=1:ndim
    for i=1:height
        for j=1:width
            feat(i,j,k) = single(feat_tmp{1}(cnt));
            cnt = cnt + 1;
        end
    end
end
for i=1:height      % set truncation dimension
    for j=1:width
        feat(i,j,ndim+1) = 0;
    end
end
  %scale = 1000*ones(size(feat));
  %feat = (feat)./scale;

fclose(fid);

%{
% this snippet directly loads the first line, which corresponds to 2X
resolution
fid = fopen(fname, 'r');

A = textscan(fid, '%f %f %f\n', 1, 'delimiter', ',');
ndim = A{1}; height = A{2}; width = A{3};

feat_tmp = textscan(fid, '%f', ndim*height*width, 'delimiter', ',');
fscanf(fid, '\n');
%feat = fscanf(fid, '%f', [ndim*height*width]);

feat = zeros(height, width, ndim);
cnt = 1;
for k=1:ndim
    for i=1:height
        for j=1:width
            feat(i,j,k) = feat_tmp{1}(cnt);
            cnt = cnt + 1;
        end
    end
end
  %scale = 1000*ones(size(feat));
  %feat = (feat)./scale;

fclose(fid);
%}

%feat_hog = features(double(imreadx(imds)), 8); %compare to hog features
%disp('here'); keyboard;

catch
    disp(lasterr); keyboard;
end
