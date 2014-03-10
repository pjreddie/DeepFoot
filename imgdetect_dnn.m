function [ds, bs, trees] = imgdetect_dnn(imds, model, thresh)
% Wrapper around gdetect.m that computes detections in an image.
%   [ds, bs, trees] = imgdetect(im, model, thresh)
%
% Return values (see gdetect.m)
%
% Arguments
%   im        Input image
%   model     Model to use for detection
%   thresh    Detection threshold (scores must be > thresh)

pyra = featpyramid_dnn(imds, model);
[ds, bs, trees] = gdetect(pyra, model, thresh);
