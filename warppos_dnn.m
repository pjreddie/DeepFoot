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
	x1 = pos(i).x1;
	x2 = pos(i).x2;
	y1 = pos(i).y1;
	y2 = pos(i).y2;
		
	scaley = cropsize(1)/(y2-y1);
	scalex = cropsize(2)/(x2-x1);

	maxscale = max(scaley, scalex);
	
	pyra = featpyramid_dnn(pos(i), model);
	orig_size = size(pyra.feat{1});
	orig_size = ([orig_size(1) orig_size(2)]- 2 .* [pyra.pady pyra.padx]) .* 4;
	index = 1
	for j = 1:size(pyra.scales,1)
		if pyra.scales(j) >= maxscale
			index = j
		end
	end
	feat = pyra.feat{index};
	scale = pyra.scales(index);
	padding = [pyra.pady+1 pyra.padx+1];

	orig_center = [(y1+y2)/2 (x1+x2)/2]
	featsize = size(feat);
	nopadsize = [featsize(1) featsize(2)]-2 .* padding;

	feat_center = orig_center .* nopadsize./orig_size;
	top_left = round(feat_center - fsize./2)+padding;
	top_left = max(top_left, padding);
	bot_right = top_left + fsize - [1 1];
	bot_right = min(bot_right, nopadsize+padding);
	top_left = bot_right - fsize + [1 1];
	
	warped{i} = feat(top_left(1):bot_right(1),top_left(2):bot_right(2),:);
	warped{i}(:,:,257) = 0 .* warped{i}(:,:,257);
	if size(warped{i}(:,:,1)) > size(nonzeros(warped{i}(:,:,1)));
		disp('**************BAD BAD BAD BAD BAD BAD********************');
	end
end

catch
	disp(lasterr); keyboard;
end
