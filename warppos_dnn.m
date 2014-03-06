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
%   pos     Positive examples from pascal_data.m

try

fi = model.symbols(model.rules{model.start}.rhs).filter;
fsize = model.filters(fi).size;
pixels = fsize * model.sbin;
heights =[pos(:).y2]' - [pos(:).y1]' + 1;
widths = [pos(:).x2]' - [pos(:).x1]' + 1;
numpos = length(pos);
warped = cell(numpos, 1);
%cropsize = (fsize+2) * model.sbin; %+2 is needed if doing operations on
%images and then calling features(); if directly indexing into extracted
%features, then not needed
cropsize = fsize * model.sbin;

parfor i = 1:numpos
    fprintf('%s %s: warp: %d/%d\n', ...
        procid(), model.class, i, numpos)
    padx = model.sbin *  widths(i) /pixels(2);
    pady = model.sbin * heights(i) /pixels(1);
    x1 = pos(i).x1;
    x2 = pos(i).x2;
    y1 = pos(i).y1;
    y2 = pos(i).y2;
    %window = subarray(im, y1, y2, x1, x2, 1);
    %im = imresize(window, cropsize, 'bilinear');
    %warped{i} = features(double(im), model.sbin);
        
    scaley = cropsize(1)/(y2-y1);
    scalex = cropsize(2)/(x2-x1);
    %fprintf('%f, %f\n', scaley, scalex);
    maxscale = max(scaley, scalex);
    
    pyra = featpyramid_dnn(pos(i), model);
	index = 1
	for j = 1:size(pyra.scales,1)
		if pyra.scales(j) >= maxscale
			index = j
		end
	end
	feat = pyra.feat{index};
	scale = pyra.scales(index);
	%fprintf('%s Index: %d, Scale: %f, Target %f\n', pos(i).im, index, pyra.scales(index), maxscale);
    %fprintf('JOE!!!: %d/%d\n', ...
    %size(feat, 1), size(feat,2));
    %fprintf('%d %d\n', fsize(1), fsize(2));
    %fprintf('box: %d,%d\n', pos(i).x2-pos(i).x1, pos(i).y2-pos(i).y1);
	%fprintf('%f %d %d\n', minscale, pos(i).x2-pos(i).x1, pos(i).y2-pos(i).y1);
	%fprintf('XB1: %f\n', scale*pos(i).x1/model.sbin);

    %feat = features_dnn(pos(i));
	featsize = size(feat);
	xb1 = ceil(scale*pos(i).x1/model.sbin)+pyra.padx+1;
	xb2 = xb1 + fsize(2)-1;
	if xb2 > featsize(2)-pyra.padx;
		xb1 = xb1 - 1;
		xb2 = xb2 - 1;
	end
	yb1 = ceil(scale*pos(i).y1/model.sbin)+pyra.pady+1;
	yb2 = yb1 + fsize(1)-1;
	if yb2 > featsize(1)-pyra.pady;
		yb1 = yb1 - 1;
		yb2 = yb2 - 1;
	end
	%fprintf('%d,%d to %d,%d out of %d x %d\n', xb1,yb1,xb2,yb2,size(feat,2), size(feat,1));
	%fprintf('%d,%d to %d,%d out of %d x %d\n', pos(i).x1, pos(i).y1, pos(i).x2, pos(i).y2, size(im, 2), size(im,1));
	%fprintf('%d,%d\n', size(feat,1), size(feat,2));
	warped{i} = feat(yb1:yb2,xb1:xb2,:);
	warped{i}(:,:,257) = 0 .* warped{i}(:,:,257);
	%fprintf('warped: %d %d\n', size(warped{i},1), size(warped{i},2));
end

catch
    disp(lasterr); keyboard;
end
