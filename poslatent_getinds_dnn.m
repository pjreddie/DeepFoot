function [compinds, scores, bboxes] ...
    = poslatent_getinds_dnn(model, pos, fg_overlap, num_fp)

try
    
conf = voc_config_dnn();
model.interval = conf.training.interval_fg;
numpos = length(pos);
pixels = model.minsize * model.sbin / 2;
minsize = prod(pixels);
fusage = zeros(model.numfilters, 1);
batchsize = max(1, 2*try_get_matlabpool_size());
num = 0;

compinds = zeros(numpos, 1);
scores = -5 * ones(numpos, 1);
bboxes = zeros(numpos, 4);

% collect positive examples in parallel batches
for i = 1:batchsize:numpos
    % do batches of detections in parallel
    thisbatchsize = batchsize - max(0, (i+batchsize-1) - numpos);
    % data for batch
    clear('data');
    empties = cell(1, thisbatchsize);
    data = struct('boxdata', empties, 'pyra', empties, 'dets', empties);
    parfor k = 1:thisbatchsize
        %for k = 1:thisbatchsize
        j = i+k-1;
        % skip small examples
        if max(pos(j).sizes) < minsize
            data(k).boxdata = cell(length(pos(j).sizes), 1);
            %fprintf('%s (all too small)\n', msg);
            continue;
        end
           
        % do whole image operations
        %im = color(imreadx(pos(j)));
        %[im, boxes, xc1, yc1] = croppos(im, pos(j).boxes);
        %[pyra, model_dp] = gdetect_pos_prepare(im, model, boxes, fg_overlap);
        pyra = featpyramid_dnn(pos(j), model);
        boxes = pos(j).boxes;
        xc1 = 1; yc1 = 1;
        [pyra, model_dp] = gdetect_pos_prepare_dnn(pyra, model, boxes, fg_overlap);
        data(k).pyra = pyra;
        
        % process each box in the image
        num_boxes = size(boxes, 1);
        for b = 1:num_boxes
            % skip small examples
            if pos(j).sizes(b) < minsize
                data(k).boxdata{b} = [];
                %fprintf('%s (%d: too small)\n', msg, b);
                continue;
            end
            fg_box = b;
            bg_boxes = 1:num_boxes;
            bg_boxes(b) = [];
            [ds, bs, trees] = gdetect_pos(data(k).pyra, model_dp, 1+num_fp, ...
                fg_box, fg_overlap, bg_boxes, 0.5);
            if isempty(ds) % added as weird aspect ratio horse was crashing (need to investigate further)
                data(k).boxdata{b} = [];
                %fprintf('%s (%d: no box detected, possibily bad aspect ratio)\n', msg, b);
                continue;
            end
            ds(:, [1 3]) = ds(:, [1 3]) + xc1 -1;
            ds(:, [2 4]) = ds(:, [2 4]) + yc1 - 1;
            
            data(k).dets{b}.dets = ds;
            data(k).boxdata{b}.bs = bs;
            data(k).boxdata{b}.trees = trees;
            %if ~isempty(bs)
            %    fprintf('%s (%d: comp %d  score %.3f)\n', msg, b, bs(1,end-1), bs(1,end));
            %else
            %    fprintf('%s (%d: no overlap)\n', msg, b);
            %end
        end
        model_dp = [];
    end
    % write feature vectors sequentially
    for k = 1:thisbatchsize
        j = i+k-1;
        myprintf(j, 10);
        num = num + 1;
        if num ~= j , disp('error'); keyboard; end
        % write feature vectors for each box
        for b = 1:length(pos(j).dataids)
            if isempty(data(k).boxdata{b})
                continue;
            end
            dataid = pos(j).dataids(b);
            bs = data(k).boxdata{b}.bs;
            ds = data(k).dets{b}.dets;
            if ~isempty(bs)                
                compinds(num) = bs(1,end-1);
                scores(num) = bs(1,end);
                bboxes(num, :) = ds(:, 1:4);
            else
                bboxes(num, :) = [pos(j).x1 pos(j).y1 pos(j).x2 pos(j).y2];
            end
        end
    end    
end
myprintfn;

catch
    disp(lasterr); keyboard;
end


function s = try_get_matlabpool_size()
try
  s = matlabpool('size');
catch
  s = 0;
end
