function H = cnn_encode(net, imdb, ids, opts)
batch_size = opts.batchSize;
if numel(opts.gpus) == 0
    onGPU = false;
else
    onGPU = true;
    %gpuDevice(opts.gpus);
end

if opts.binary
    fprintf('Testing network [%s] -> BINARY descriptors\n', opts.arch);
elseif opts.l2norm
    fprintf('Testing network [%s] -> REAL (l2normd) descriptors\n', opts.arch);
else
    fprintf('Testing network [%s] -> REAL (sigmoid) descriptors\n', opts.arch);
end

% prepare network
if isa(net, 'dagnn.DagNN')
    net.mode = 'test';
    net.layers(end).block.aff_aux = [];
    if opts.sqdist
        ind = net.getVarIndex('feats_l2');
    else
        ind = net.getVarIndex('logits');
    end
    if isempty(ind)
        ind = net.getVarIndex('bn7');
    end
    net.vars(ind).precious = 1;
    if onGPU, net.move('gpu'); end
else
    net.layers(end) = [];
    if onGPU, net = vl_simplenn_move(net, 'gpu'); end
end

% process input data in batches
H = zeros(opts.nbits, length(ids), 'single');
tic;
for t = 1:batch_size:length(ids)
    % NOTE: this assumes that the input patches has been properly normalized and 
    %       stored in an 'imdb' structure (MatConvNet convention).
    %
    %       Otherwise, use 'patch_normalize.m' to normalize raw input patches.
    %
    ed = min(t+batch_size-1, length(ids));
    data  = imdb.images.data(:, :, :, ids(t:ed));
    label = imdb.images.labels(ids(t:ed), :);

    if onGPU, data = gpuArray(data); end
    if isa(net, 'dagnn.DagNN')
        net.eval({'input', data, 'labels', label});
        rex = net.vars(ind).value;
    else
        net.layers{end}.class = label;
        res = vl_simplenn(net, data, [], [], ...
            'mode', 'test', 'cudnn', true, 'conserveMemory', true);
        rex = res(end).x;
    end
    rex = squeeze(gather(rex));
    H(:, t:ed) = rex;
end
if onGPU && isa(net, 'dagnn.DagNN')
    net.reset(); 
    net.move('cpu'); 
end

% post-processing
if opts.binary
    H = (H > 0);
elseif opts.l2norm && ~opts.sqdist
    H = bsxfun(@rdivide, H, sqrt(sum(H.^2, 1)));
else
    H = sigmoid(H, opts.sigmf);
end
toc;
end
