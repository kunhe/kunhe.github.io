function P = patch_normalize(P)
% input: 
% P - W x H x 1 x N  (WxH size, grayscale, N total)
for i = 1:size(P, 4)
    Pi = P(:, :, :, i);
    P(:, :, :, i) = (Pi - mean(Pi(:))) ./ std(Pi(:));
end
end
