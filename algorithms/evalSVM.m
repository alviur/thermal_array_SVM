%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Alexander Gomez Villa - Jefferson Cunalata - Fabio Arnez
% -------------------------------------------------------------------------
% predict SVM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function prediction = evalSVM(supVect,supVectLabels,weigths,input, kernel)

    [r1 c1] = size(supVect);
    [r2 c2] = size(input);
    kpar1 = 0.001;

    % Select kernel
    switch kernel
        case 'rbf'
            k = zeros(r1,1);
            for i = 1 : r1
                k(i) = exp(-(supVect(i,:)-input)*(supVect(i,:)-input)'/(2*kpar1^2));
            end

    end

    t=sum((supVect.*supVectLabels).*k')-w0;
    if(t>0)
        prediction=1;
    else
        prediction=-1;
    end

end