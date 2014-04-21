function pyra = featpyramid_dnn(imds, model, padx, pady)
% function that takes an image name and loads its full DNN feature pyramid

try
    
% hardcoded path
pathToFeatures = '/projects/grail/unikitty/VOC2007/imagenet_features/';
if model.interval == 4
pathToFeatures = '/projects/grail/unikitty/VOC2007/imagenet_features/4/';
end
if model.interval == 10
pathToFeatures = '/projects/grail/unikitty/VOC2007/imagenet_features/10/';
end

% imds has 'im', 'flip', 'boxes' (if positive); see pascal_data.m
alltoks = strtokAll(imds.im, '/');
fname = [pathToFeatures '/' alltoks{end} '.txt'];
% flip features if needed
if imds.flip
    fname = [pathToFeatures '/' alltoks{end} '_r.txt'];
end

%%% Read data from file
fid = fopen(fname, 'r');
fcnt = 1;
clear feats;
while ~feof(fid)
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
    
    feats{fcnt} = feat;
    fcnt = fcnt + 1;
end
fclose(fid);

%sampint = round(10/model.interval);
%feats = feats{1:sampint:end};

if nargin < 3
  [padx, pady] = getpadding(model);
end
sbin = model.sbin;
interval = model.interval;
sc = 2^(1/interval);
im = double(imreadx(imds));
imsize = [size(im, 1) size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));
pyra.feat = cell(max_scale + interval, 1);
pyra.scales = zeros(max_scale + interval, 1);
pyra.imsize = imsize;
for i = 1:interval
    % (sbin/2) x (sbin/2) features
    pyra.feat{i} = feats{i};
    pyra.scales(i) = 2/sc^(i-1);
    % sbin x sbin HOG features
    pyra.feat{i+interval} = feats{i+interval};
    pyra.scales(i+interval) = 1/sc^(i-1);
    % Remaining pyramid octaves
    for j = i+interval:interval:max_scale        
        pyra.feat{j+interval} = feats{j+interval};
        pyra.scales(j+interval) = 0.5 * pyra.scales(j);
    end
end

pyra.num_levels = length(pyra.feat);

td = model.features.truncation_dim;
for i = 1:pyra.num_levels
  % add 1 to padding because feature generation deletes a 1-cell
  % wide border around the feature map

  scale = 70.0*ones(size(pyra.feat{i}));
  pyra.feat{i} = (pyra.feat{i})./scale;
  pyra.feat{i} = padarray(pyra.feat{i}, [pady+1 padx+1 0], 0);
  % write boundary occlusion feature
  pyra.feat{i}(1:pady+1, :, td) = 1;
  pyra.feat{i}(end-pady:end, :, td) = 1;
  pyra.feat{i}(:, 1:padx+1, td) = 1;
  pyra.feat{i}(:, end-padx:end, td) = 1;
end
pyra.valid_levels = true(pyra.num_levels, 1);
pyra.padx = padx;
pyra.pady = pady;

%pyra_hog = featpyramid((imreadx(imds)), model);

catch
    disp(lasterr); keyboard;
end
 
