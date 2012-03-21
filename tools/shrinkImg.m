% shrink image size according to requirment
function shrinkedImg = shrinkImg (img, rs)
        [r,c]=size(img);
        shrinkedImg = img(1:rs:r , 1:rs:c );
endfunction

