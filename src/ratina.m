close all;
I = (imread('im0163.ppm'));
I = I(:, :, 2);

J = adapthisteq(I);

se = strel('disk', 12);
If = imtophat(J, se); %try changing second argument

Ir = (If < 3);
figure;
imshow(Ir, []);
figure;

[n,m] = size(Ir);
% J = J(80:n-80,80:m-80);
[n,m] = size(J);

Ir = -0.1*double(J);
imshow(J, []);
figure;


I_new = zeros(n,m);

for i = 2:n-1
    for j = 2:m-1
       I_new(i,j) = -0.35*Ir(i,j-1) + Ir(i,j) - 0.35*Ir(i,j+1);
    end
end
imshow(I_new, []);
figure;

for i = 1:n
    x = mean(I_new(i,:));
    sig = std(I_new(i,:));
    alpha = 0.0033;
    th = x + sig*alpha/(rms(I_new(i,:))^2);
    for j = 1:m
        if I_new(i,j) > th
            I_new(i,j) = 1;
        else
            I_new(i,j)= 0;
        end
    end
end

imshow(I_new, []);
