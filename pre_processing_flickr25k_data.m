function pre_processing_flickr25k_data(path_to_data)
    n_files = 25000;
    disp('Start processing flickr25k images');
    tic;
    for i = 1 : n_files
        filename = [path_to_data, 'im', num2str(i), '.jpg'];
        imgRGB = imread(filename);
        imgGray = rgb2gray(imgRGB);
        imgData_H = imresize(imgGray, [256, 256]);
        imgData_L = imresize(imgGray, [64, 64]);
        imgData_L = imresize(imgData_L, [256, 256]);
        imwrite(imgData_H, ['imh',num2str(i),'.jpg']);
        imwrite(imgData_L, ['iml',num2str(i),'.jpg']);
        if(mod(i, 100) == 0)
            t_cost = toc;
            t_remain = (n_files/i - 1)*t_cost;
            disp(['Processed ',num2str(i),' images with ', num2str(t_remain), ' seconds remain.']);
        end
    end
end