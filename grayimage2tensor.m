function grayimage2tensor(data_path, prefix, id_list, suffix, n_img_per_tensor)
%IMAGE2TENSOR Summary of this function goes here
%   Detailed explanation goes here
file_name_tmp = [data_path, prefix,num2str(id_list(1)),suffix];
imgtemp = imread(file_name_tmp);
[height, width] = size(imgtemp);
total_file = numel(id_list);
n_file = ceil(total_file/n_img_per_tensor);
tic;
for i = 1 : n_file
    id0 = (i-1)*n_img_per_tensor + 1;
    id1 = id0 + n_img_per_tensor - 1;
    id1 = min(id1, total_file);
    id_list_file = id_list(id0:id1);
    data = zeros(n_img_per_tensor, height, width, 1);
    for j = 1 : numel(id_list_file)
        id_img = id_list_file(j);
        file_name_tmp = [data_path, prefix,num2str(id_img),suffix];
        imgtmp = imread(file_name_tmp);
        imgtmp = double(imgtmp)/255;
        data(j, :, :, 1) = imgtmp;
    end
    filename_out = [data_path, prefix, num2str(i), '.mat'];
    save(filename_out,'data');
    t_cost = toc;
    t_remain = (n_file/i - 1)*t_cost;
    disp(['Processed ',num2str(id_list_file(end)),' images with ', num2str(t_remain), ' seconds remain.']);    
end
end

