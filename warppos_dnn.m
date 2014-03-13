function warped = warppos_dnn(model, pos)
% Warp positive examples to fit model dimensions.
%   warped = warppos(model, pos)
%
%   Used for training root filters from positive bounding boxes.
%
% Return value
%   warped  Cell array of images
%
% Arguments
%   model   Root filter only model
%   pos	 Positive examples from pascal_data.m

try

fi = model.symbols(model.rules{model.start}.rhs).filter;
fsize = model.filters(fi).size;
pixels = fsize * model.sbin;
heights =[pos(:).y2]' - [pos(:).y1]' + 1;
widths = [pos(:).x2]' - [pos(:).x1]' + 1;
numpos = length(pos);
warped = cell(numpos, 1);
cropsize = (fsize+2) * model.sbin; %+2 is needed if doing operations on
%images and then calling features(); if directly indexing into extracted
%features, then not needed
%cropsize = fsize * model.sbin;

parfor i = 1:numpos
	fprintf('%s %s: warp: %d/%d\n', ...
		procid(), model.class, i, numpos)
    height = fsize(1);
    width = fsize(2);
    ndim = 257;
	x1 = pos(i).x1;
	x2 = pos(i).x2;
	y1 = pos(i).y1;
	y2 = pos(i).y2;
		
    im = imreadx(pos(i));
    window = subarray(im, y1, y2, x1, x2, 1);

    alltoks = strtokAll(pos(i).im, '/');

    filename = sprintf('/tmp/%d%s', i, alltoks{end});
    imwrite(window, filename);
    command = sprintf('bash --login -c "./jnet/cnn %s %d %d 2>/dev/null"', filename, fsize(1), fsize(2));
    [status res] = system(command);
    feat_tmp = textscan(res, '%f', height*width*(ndim-1), 'delimiter', ',');

    feat = zeros(height, width, ndim, 'single');  % +1 is for truncation dim
    cnt = 1;
    for k=1:ndim-1
        for ii=1:height
            for j=1:width
                feat(ii,j,k) = single(feat_tmp{1}(cnt));
                cnt = cnt + 1;
            end
        end
    end
    scale = 1000*ones(size(feat));
    feat = feat./scale;
	
	warped{i} = feat;
end

catch
	disp(lasterr); keyboard;
end
