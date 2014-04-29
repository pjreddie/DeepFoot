function masterScript_dnn(OBJINDS)

% masterScript to run the dnn feats on VOC dataset
% author: Santosh Divvala (santosh@cs.washington.edu), 2014

% OBJINDS: optional argument to indicate the index of a particular concept to run (see VOCoptsClasses.m for list)

[allClasses, ~, POStags] = VOCoptsClasses;      % get list of all concepts and their corresponding parts-of-speech (POS) tags
if ~exist('OBJINDS', 'var') || isempty(OBJINDS)
    OBJINDS = numel(allClasses):numel(allClasses); 
end 

%%% data year and types
trainyear = '2007';                         % what year of data to train on (default value: '9990' => web data)
testyear = '2007'; testdatatype = 'test';    % what year,type of data to test the models on ('2007, test', '2011, val',...)

%%% main path names
imgannodir = ['/projects/grail/santosh/Datasets/Pascal_VOC/VOC2007/']; % path to the voc 2007 data (whose images are used as background/negative data for training all models)
jpgimagedir = [imgannodir '/JPEGImages/']; mymkdir(jpgimagedir);
imgsetdir = [imgannodir '/ImageSets/Main/']; mymkdir(imgsetdir);
imgsetdir_voc = [imgannodir '/ImageSets/voc/']; mymkdir(imgsetdir_voc);
annosetdir = [imgannodir '/Annotations/']; mymkdir(annosetdir);

basedir = ['/projects/grail/unikitty/objectNgrams/'];                 % main project folder (with the code, results, etc)
resultsdir = fullfile(basedir, 'results', 'dpmWithDNN_updates');

%%% global variables (need to put them here instead of voc_config.m)
OVERWRITE = 1;                      % whether to overwrite compiled code or not
dpm.numcomp = 6;                    % number of components for training DPM
dpm.wsup_fg_olap = 0.5;             % amount of foreground overlap (with ground-truth bbox)
dpm.jointCacheLimit = 2*(3*2^30);   % amount of RAM for training DPM

%%% main code
for objind = OBJINDS            % run either all concepts or a selected concept
    
    objname = allClasses{objind};   
    thisPOStag = POStags{objind};   
    
    % set all the path names for this concept   
    ngramModeldir_obj = [resultsdir '/' objname '/']; mymkdir(ngramModeldir_obj); % to save data/results for DPM 
    
    diary([resultsdir '/' objname '/diaryOutput_all.txt']);        % save a log of the entire run for debugging/record purposes
    
    disp(['Doing base object category ' num2str(objind) '.' objname]);
        
    
    disp('%%% DPM TRAINING');    
    %compileCode_v2_depfun('pascal_train_wsup3', 1);
    doparts = 0;
    modelname = 'mix';
    cachedir = [ngramModeldir_obj '/']; mymkdir(cachedir);
    if ~exist([cachedir '/' objname '_' modelname '.mat'], 'file')
        pascal_train_dnn(objname, dpm.numcomp, 'blah', cachedir, trainyear, dpm.wsup_fg_olap, doparts);
        %pascal_train(objname, dpm.numcomp, 'blah', cachedir, trainyear, dpm.wsup_fg_olap, doparts);
        %multimachine_grail_compiled(['pascal_train_dnn ' objname ' ' num2str(dpm.numcomp) ' ' 'blah' ' ' cachedir  ' ' trainyear ' ' num2str(dpm.wsup_fg_olap) ' ' num2str(doparts)], 1, cachedir, 1, [], 'all.q', 8, 0, OVERWRITE);
    end
            
    
    disp('%%% TESTING (on voc data)');
    %compileCode_v2_depfun('pascal_test_sumpool_multi', 1, 'linuxUpdateSystemNumThreadsToMax.sh');
    compileCode_v2_depfun('pascal_test_sumpool_multi_dnn', 1, 'linuxUpdateSystemNumThreadsToMax.sh');
    if exist([cachedir '/' objname '_' modelname '.mat'], 'file') &&...
            ~exist([cachedir '/' objname '_boxes_' testdatatype '_' testyear '_' modelname '.mat'], 'file') % test _joint model on test set (cluster version)
        resdir = [cachedir '/testFiles_' testyear '/'];
        num_ids = getNumImagesInDataset(cachedir, testyear, testdatatype, thisPOStag);
        if areAllFilesDone(resdir, num_ids, [], 1) ~= 0            
            %pascal_test_sumpool(cachedir, objname, testdatatype, testyear, testyear, modelname);                     % single machine version            
            %pascal_test_sumpool_multi_dnn(cachedir, objname, testdatatype, testyear, testyear, modelname, thisPOStag);      % cluster version
            numjobsDetTest = min(200, areAllFilesDone(resdir, num_ids, [], 1));
multimachine_grail_compiled(['pascal_test_sumpool_multi_dnn ' cachedir ' ' objname ' ' testdatatype ' ' testyear ' ' testyear ' ' modelname ' ' thisPOStag], num_ids, resdir, numjobsDetTest, [],'notcuda.q', 1, 0, OVERWRITE, 0);
            %multimachine_grail_compiled(['pascal_test_sumpool_multi ' cachedir ' ' objname ' ' testdatatype ' ' testyear ' ' testyear ' ' modelname ' ' thisPOStag], num_ids, resdir, numjobsDetTest, [], 'all.q', 8, 0, OVERWRITE, 0);
            areAllFilesDone(resdir, num_ids);
        end
        pascal_test_sumpool_reducer(cachedir, objname, testdatatype, testyear, testyear, modelname, thisPOStag);
    end
    myprintfn; myprintfn;
            
    
    disp('%%% EVALUATION (when Ground-truth available)');
    if ~exist([cachedir '/' objname '_pr_' testdatatype '_' [testyear '_' modelname '_' num2str(100*0.5)] '.mat'], 'file')
        pascal_eval_ngramEvalObj(objname, objname, cachedir, testdatatype, testyear, [testyear '_' modelname], 0.5, thisPOStag);
    end    
    myprintfn; myprintfn;
    
                
    diary off;
end
