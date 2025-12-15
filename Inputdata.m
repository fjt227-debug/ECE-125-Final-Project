data=out.y;

x1=data(:,1);
x2=data(:,2);
x3=data(:,3);
x4=data(:,4);
u=data(:,5);
y=data(:,6);

X=[x1,x2,x3,x4,u]; %%training data
Y=y; %%label

save('traingDATA.mat','X','Y');

k = 1:length(y); %sample index

% Figure 1:x1
figure;
plot(k, x1, 'LineWidth', 1.5);
xlabel('Sample Time S');
ylabel('x1(k)');
title('x1');
grid on;

% Figure 2:x2
figure;
plot(k, x2, 'LineWidth', 1.5);
xlabel('Sample Time S');
ylabel('x2(k)');
title('x2');
grid on;

% Figure 3:x3
figure;
plot(k, x3, 'LineWidth', 1.5);
xlabel('Sample Time S');
ylabel('x3(k)');
title('x3');
grid on;

% Figure 4:x4
figure;
plot(k, x4, 'LineWidth', 1.5);
xlabel('Sample Time S');
ylabel('x4(k)');
title('x4');
grid on;

% Figure 5:u
figure;
plot(k, u, 'LineWidth', 1.5);
xlabel('Sample Time S');
ylabel('u(k)');
title('u');
grid on;

% Figure 6:y
figure;
plot(k, y, 'LineWidth', 1.5);
xlabel('Sample Time S');
ylabel('y(k)');
title('Output y');
grid on;