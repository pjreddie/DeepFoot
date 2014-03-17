function model = pascal_train_dnn(cls, n, note, cachedir, year, fg_olap, doparts)
% new idea that trains a model partially, tests on val set, ignore bad
% comps and then trains model fully

try
% At every "checkpoint" in the training process the 
% RNG's seed is reset to a fixed value so that experimental results are 
% reproducible.
seed_rand();

if isdeployed, n = str2num(n); end
if isdeployed, fg_olap = str2num(fg_olap); end
if isdeployed, doparts = str2num(doparts); end

global VOC_CONFIG_OVERRIDE;
VOC_CONFIG_OVERRIDE.paths.model_dir = cachedir;
VOC_CONFIG_OVERRIDE.pascal.year = year;
VOC_CONFIG_OVERRIDE.training.fg_overlap = fg_olap; %0.25;
VOC_CONFIG_OVERRIDE.training.train_set_fg = 'train';
diary([cachedir '/diaryoutput_train.txt']);
disp(['pascal_train_dnn(''' cls ''',' num2str(n) ',''' note ''',''' cachedir ''',''' year ''',' num2str(fg_olap) ','  num2str(doparts) ')' ]);

disp(' only opening 12 cores'); 
mymatlabpoolopen(12);

conf = voc_config_dnn();
save([cachedir cls '_conf.mat'], 'conf');

% Load the training data
[pos, neg, impos] = pascal_data(cls, conf.pascal.year);

disp('splitting - kmeans using esvm hog'); 
indsname = [cachedir '/' cls '_displayInfo.mat'];
if ~exist(indsname, 'file')
    disp('doing clustering');        
    [inds_init, clustCents, mimg] = split_app(pos, n);
    save(indsname, 'inds_init', 'clustCents');
    mymkdir([cachedir '/display/']);
    imwrite(mimg, [cachedir '/display/initmontage_kmeansHOG_' num2str(n) '.jpg']);    
else
    load(indsname, 'inds_init');
end
spos = cell(n,1);
for i=1:n, spos{i} = pos(inds_init == i); end
disp(spos);

mymkdir([cachedir '/intermediateModels/']); 

max_num_examples = conf.training.cache_example_limit;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Select a small, random subset of negative images
% All data mining iterations use this subset, except in a final
% round of data mining where the model is exposed to all negative
% images
num_neg   = length(neg);
neg_perm  = neg(randperm(num_neg));
neg_small = neg_perm(1:min(num_neg, conf.training.num_negatives_small));
neg_large = neg;        % use all of the negative images

% Train a root filter for each subcategory
% using warped positives and random negatives
disp('Doing lrsplit1');
try
    load([cachedir cls '_lrsplit1']);
catch
    seed_rand();
    for i = 1:n
        disp(['*******Training lrsplit1 model ' num2str(i) ' ********']);
        models{i} = root_model_dnn(cls, spos{i}, note);                
        models{i} = train_dnn(models{i}, spos{i}, neg_large, true, true, 1, 1, ...
            max_num_examples, fg_overlap, 0, false, ...
            ['lrsplit1_' num2str(i)]);
    end
    save([cachedir cls '_lrsplit1'], 'models');
    
    [inds_lrsplit1, posscores_lrsplit1, lbbox_lrsplit1] = poslatent_getinds_dnn(model_merge(models), pos, fg_overlap, 0);
    save([cachedir cls '_lrsplit1'], 'inds_lrsplit1', 'posscores_lrsplit1', 'lbbox_lrsplit1', '-append');
end
myprintfn;

%{
%%% debugging code
[mimg_lrs1, mlab_lrs1] = getMontagesForModel_latent_wsup(inds_lrsplit1, inds_lrsplit1, ...
    inds_lrsplit1, posscores_lrsplit1, posscores_lrsplit1, lbbox_lrsplit1, pos, [], n);
mimg = montage_list_w_text(mimg_lrs1, mlab_lrs1, 2, '', [0 0 0], [2000 2000 3]);
imwrite(mimg, [cachedir '/display/montage_lrsplit1c.jpg']);
%}

% Train a mixture model composed of all subcategories 
% using latent positives and hard negatives
disp('Doing mix');
try 
  load([cachedir cls '_mix']);
catch
  seed_rand();  
  model = model_merge(models);      % Combine separate mixture models into one mixture model
  model = train_dnn(model, impos, neg_small, false, false, 1, 100, ...
      max_num_examples, fg_overlap, num_fp, false, 'mix_1');
  model_mix1 = model;
  %model = train_wsup(model, impos, neg_large, false, false, 1, 15, ...
  %    max_num_examples, fg_overlap, num_fp, true, 'mix_2');      
  
  save([cachedir cls '_mix'], 'model', 'model_mix1');
  
  [inds_mix, posscores_mix, lbbox_mix] = poslatent_getinds_dnn(model, pos, fg_overlap, 0);
  save([cachedir cls '_mix'], 'inds_mix', 'posscores_mix', 'lbbox_mix', '-append');
  
  displayExamplesPerSubcat4(cls, cachedir, year, conf.training.train_set_fg);
end
myprintfn;
    
%{
%%% debugging code
[mimg_lrs1, mlab_lrs1] = getMontagesForModel_latent_wsup(inds_mix, inds_mix, ...
    inds_mix, posscores_mix, posscores_mix, lbbox_mix, pos, [], n);
mimg = montage_list_w_text(mimg_lrs1, mlab_lrs1, 2, '', [0 0 0], [2000 2000 3]);
imwrite(mimg, [cachedir '/display/montage_mix.jpg']);
%}

if doparts
    % Train a mixture model with 2x resolution parts using latent positives and hard negatives    
    disp('Doing parts');
    try        
        load([cachedir cls '_parts'], 'model'); model;        
    catch
        seed_rand();
        
        % Add parts to each mixture component
        for i = 1:n
            % Top-level rule for this component
            ruleind = i;
            % Top-level rule for this component's mirror image
            partner = [];
            % Filter to interoplate parts from
            filterind = i;                        
            model = model_add_parts(model, model.start, ruleind, ...
                partner, filterind, 8, [6 6], 1);
            % Enable learning location/scale prior
            bl = model.rules{model.start}(i).loc.blocklabel;
            model.blocks(bl).w(:)     = 0;
            model.blocks(bl).learn    = 1;
            model.blocks(bl).reg_mult = 1;
        end
        
        % Train using several rounds of positive latent relabeling
        % and data mining on the small set of negative images
        model = train_dnn(model, impos, neg_small, false, false, 8, 10, ...
            max_num_examples, fg_overlap, num_fp, false, 'parts_1');
        % Finish training by data mining on all of the negative images
        model = train_dnn(model, impos, neg_large, false, false, 1, 100, ...
            max_num_examples, fg_overlap, num_fp, true, 'parts_2');
        save([cachedir cls '_parts'], 'model');
        
        [inds_parts, posscores_parts, lbbox_parts] = poslatent_getinds_dnn(model, pos, fg_overlap, 0);
        save([cachedir cls '_parts'], 'inds_parts', 'posscores_parts', 'lbbox_parts', '-append');        
        
        displayExamplesPerSubcat4(cls, cachedir, year, conf.training.train_set_fg);
    end
end

fv_cache('free');

save([cachedir cls '_final'], 'model');

%displayWeightVectorsPerAspect_v5(cls, cachedir);
close all;
diary off;

catch
    disp(lasterr); keyboard;
end
