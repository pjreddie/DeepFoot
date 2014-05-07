function viz1(id, cls, testset, cachedir, year, suffix, modelname)
%VIZ1 Summary of this function goes here
%   Detailed explanation goes here

global VOC_CONFIG_OVERRIDE;
%VOC_CONFIG_OVERRIDE = @my_voc_config_override;
VOC_CONFIG_OVERRIDE.paths.model_dir = cachedir;
VOC_CONFIG_OVERRIDE.pascal.year = year;

conf = voc_config('paths.model_dir', cachedir, 'pascal.year', year, 'eval.test_set', testset);
VOCopts  = conf.pascal.VOCopts;

if exist('/home/ubuntu/JPEGImages/','dir')  % for aws
    VOCopts.imgpath = '/home/ubuntu/JPEGImages/%s.jpg';
end

detressavedir = [cachedir '/display_' testset '_' year '_' suffix '/']; mymkdir(detressavedir);

ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');
load([cachedir 'horse_boxes_test_2007_mix.mat']);
im = imread(['/projects/grail/santosh/Datasets/Pascal_VOC/VOC2007/JPEGImages/' ids{id} '.jpg']);
showboxes(im, ds{id});

end

