function [bestIdx, bestclusCentMat, mimg] = split_app_dnn(pos, numClustersPerSplit)
% this script is not being used

try

fprintf(' warping instances\n');
model = root_model('blah', pos, []);
%model = root_model('blah', pos, [], 8, [12 12]);
warped = warppos(model, pos);

fprintf(' caching features\n');
feats = cell(length(warped),1);
sbinval = model.sbin;
%for i = 1:length(warped)
parfor i = 1:length(warped)
    myprintf(i,100);    
    feats{i} = features(double(warped{i}), sbinval);
    feats{i} = feats{i}(:);
end
myprintfn;
featMat = cat(2,feats{:})';

%maxiter = 2;
maxiter = 5;    % changed it back to 5 as I have uncanny feeling that results would be hurt and then this is just one time investment
disp([' doing clustering' num2str(maxiter)]);
bestv = inf; 
%{
for j = 1:maxiter
    myprintf(j);
            
    if numel(pos) < 4000  % plug added to deal with 'person', 6May12
        [Idx, clusCentMat, v] = kmeans(featMat, numClustersPerSplit, 'Replicates', 5);
    else
        [Idx, clusCentMat, v] = kmeanspp(featMat', numClustersPerSplit);
        Idx = Idx(:); clusCentMat = clusCentMat';
    end
        
    v = sum(v);
    if v < bestv
        fprintf('new total intra-cluster variance: %f\n', v);
        bestv = v;
        bestIdx = Idx';
        bestclusCentMat = clusCentMat';
    end
end
myprintfn;
%}
[pf_Idx, pf_clusCentMat, pf_v] = deal(cell(numel(maxiter),1));
parfor j = 1:maxiter
    myprintf(j);
            
    if numel(pos) < 4000  % plug added to deal with 'person', 6May12
        [pf_Idx{j}, pf_clusCentMat{j}, pf_v{j}] = kmeans(featMat, numClustersPerSplit, 'Replicates', 5);
    else
        [pf_Idx{j}, pf_clusCentMat{j}, pf_v{j}] = kmeanspp(featMat', numClustersPerSplit);
        pf_Idx{j} = pf_Idx{j}(:); pf_clusCentMat{j} = pf_clusCentMat{j}';
    end
end

for j=1:maxiter
    v = pf_v{j};
    v = sum(v);
    if v < bestv
        fprintf('new total intra-cluster variance: %f\n', v);       
        bestv = v;
        bestIdx = pf_Idx{j}';
        bestclusCentMat = pf_clusCentMat{j}';
    end
end
myprintfn;

%sid1 = tic;
try
% visualize
[mimg_lrs, mlab_lrs] = getMontagesForModel_latent(bestIdx(:), bestIdx(:), ...
    bestIdx(:), [], [], [], pos, [], numClustersPerSplit);  
mimg = montage_list_w_text2(mimg_lrs, mlab_lrs, 2, '', [0 0 0], [2000 2000 3]);
end
%disp('visualize for clustering took');
%toc(sid1);

catch
    disp(lasterr); keyboard;
end
 