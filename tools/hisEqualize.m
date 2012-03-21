% function to first normalize the input img to double format with value 0 to 1.0 and then perform histogram transformation
function histedImg = hisEqualize(img)
        doubleImg = im2double(img);
 	histedImg = histeq(doubleImg, 256);
endfunction

