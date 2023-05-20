function outputImage = medianFilter(inputImage, windowSize)
    % Get the size of the input image
    [rows, cols] = size(inputImage);
    
    % Calculate the padding size based on the window size
    paddingSize = floor(windowSize / 2);
    
    % Create a padded version of the input image
    paddedImage = padarray(inputImage, [paddingSize, paddingSize]);
    
    % Create an output image with the same size as the input image
    outputImage = zeros(rows, cols);
    
    % Apply the median filter
    for i = 1:rows
        for j = 1:cols
            % Extract the window from the padded image
            window = paddedImage(i:i+windowSize-1, j:j+windowSize-1);
            
            % Calculate the median value of the window
            medianValue = median(window(:));
            
            % Set the output pixel value to the median value
            outputImage(i, j) = medianValue;
        end
    end
    %outputImage = uint8(outputImage);
end
