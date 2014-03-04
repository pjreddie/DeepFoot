function pascal_test_sumpool_multi_dnn(cachedir, cls, testset, year, suffix, modelname, postag)
% Compute bounding boxes in a test set.

try    

global VOC_CONFIG_OVERRIDE;
%VOC_CONFIG_OVERRIDE = @my_voc_config_override;
VOC_CONFIG_OVERRIDE.paths.model_dir = cachedir;
VOC_CONFIG_OVERRIDE.pascal.year = year;

if nargin < 6
    modelname = '';
end

if nargin < 7
    postag = 'NOUN';
end

disp(['pascal_test_sumpool_multi_dnn(''' cachedir ''',''' cls ''',''' testset ''',''' year ''',''' suffix ''',''' modelname ''',''' postag ''');' ]);

filenameWithPath = which('linuxUpdateSystemNumThreadsToMax.sh');    % avoids hardcoding filepath (/projects/grail/santosh/objectNgrams/code/utilScripts/linuxUpdateSystemNumThreadsToMax.sh')
system(['. ' filenameWithPath]);

conf = voc_config_dnn('pascal.year', year, 'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;
cachedir = conf.paths.model_dir;

% copied from pascal.m
if isempty(modelname)   % if modelname = 'final', leave it empty
    disp('loading final (parts) model');
    load([cachedir '/' cls '_final.mat'], 'model');
    %savename = [cachedir cls '_boxes_' testset '_' suffix];
else
    disp('loading non-final (no parts/mix/joint) model');
    load([cachedir '/' cls '_' modelname '.mat'], 'model');
    %savename = [cachedir cls '_boxes_' testset '_' suffix '_' modelname];
end
resdir = [cachedir '/testFiles_' year '/']; mymkdir(resdir);

model.thresh = min(conf.eval.max_thresh, model.thresh);
model.interval = conf.eval.interval;

if strcmp(postag, 'NOUN')
    ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');
elseif strcmp(postag, 'VERB')
    ids = textread(sprintf(VOCopts.action.imgsetpath, testset), '%s');
end

% parfor gets confused if we use VOCopt
opts = VOCopts;
if exist('/home/ubuntu/JPEGImages/','dir')  % for aws
    disp('updating image path /home/ubuntu/JPEGImages');
    opts.imgpath = '/home/ubuntu/JPEGImages/%s.jpg'; 
end
num_ids = length(ids);

%for f = 1:num_ids
mymkdir([resdir '/done']);
myRandomize;
list_of_ims = randperm(num_ids); 
for f = list_of_ims
    if (exist([resdir '/done/' num2str(f) '.lock'],'dir') || exist([ resdir '/done/' num2str(f) '.done'],'dir') )
        continue;
    end
    if mymkdir_dist([resdir '/done/' num2str(f) '.lock']) == 0
        continue;
    end
        
    disp(['Processing image ' num2str(f)]);
    
    fname = [resdir '/output_' num2str(f) '.mat'];
    try
        load(fname, 'ds_save', 'bs_save', 'ds_sumsave');
        ds_save;
    catch        
        if strcmp('inriaperson', cls)
            % INRIA uses a mixutre of PNGs and JPGs, so we need to use the annotation
            % to locate the image.  The annotation is not generally available for PASCAL
            % test data (e.g., 2009 test), so this method can fail for PASCAL.
            rec = PASreadrecord(sprintf(opts.annopath, ids{f}));
            im = imread([opts.datadir rec.imgname]);
        else
            im = imread(sprintf(opts.imgpath, ids{f}));
        end
        
        imds.im = sprintf(opts.imgpath, ids{f});
        imds.flip = 0;
        [ds, bs] = imgdetect_dnn(imds, model, model.thresh);                   
               
        if ~isempty(bs)
            unclipped_ds = ds(:,1:4);
            [ds, bs, rm] = clipboxes(im, ds, bs);
            unclipped_ds(rm,:) = [];
            
            % sumpooling
            ds_sum = ds;
            ds_sum = decodeDets(ds_sum);
            I = nms(ds_sum, 0.5);
            ds_sum = ds_sum(I,:);
            
            % NMS
            I = nms(ds, 0.5);
            ds = ds(I,:);
            bs = bs(I,:);
            unclipped_ds = unclipped_ds(I,:);
            
            % Save detection windows in boxes
            ds_save = ds(:,[1:4 end]);
            ds_sumsave = ds_sum(:,[1:4 end]); % sumpooling
            
            % Save filter boxes in parts
            if model.type == model_types.MixStar
                % Use the structure of a mixture of star models
                % (with a fixed number of parts) to reduce the
                % size of the bounding box matrix
                bs = reduceboxes(model, bs);
                bs_save = bs;
            else
                % We cannot apply reduceboxes to a general grammar model
                % Record unclipped detection window and all filter boxes
                bs_save = cat(2, unclipped_ds, bs);
            end
        else
            ds_save = [];
            ds_sumsave = [];
            bs_save = [];
        end        
        save(fname, 'ds_save', 'bs_save', 'ds_sumsave');
    end
    mymkdir([resdir '/done/' num2str(f) '.done'])
    rmdir([resdir '/done/' num2str(f) '.lock']);
end

catch
    disp(lasterr); keyboard;
end
