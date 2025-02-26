clc; clear; close all; tic;

% Load city coordinates
cC = load('dj44.txt');
numCities = size(cC, 1);
x = cC(1:numCities, 2);
y = cC(1:numCities, 3);
x(numCities+1) = cC(1, 2);
y(numCities+1) = cC(1, 3); 

% Plot city coordinates
figure;
plot(x', y', '.k', 'MarkerSize', 14);
labels = cellstr(num2str([1:numCities]'));
text(x(1:numCities)', y(1:numCities)', labels, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
ylabel('Y Coordinate', 'fontsize', 18, 'fontname', 'Arial');
xlabel('X Coordinate', 'fontsize', 18, 'fontname', 'Arial');
title('City Coordinates', 'fontsize', 20, 'fontname', 'Arial');
hold on;

% Initializations
cityRoute_i = nearestNeighbor(cC, numCities); % Use for initial route
cityRoute_b = cityRoute_i;
cityRoute_o = cityRoute_i;

% Initial distances
D_j = computeEUCDistance(numCities, cC, cityRoute_i);
D_o = D_j; D_b = D_j; D(1) = D_j;
numAcceptedSolutions = 1.0;

% Parameters
numCoolingLoops = 500;
Threshold = zeros(numCoolingLoops, 1);
th1 = 0.3;
th2 = 0.00001;
thCurrent = th1;
mul1 = nthroot(th2/th1, numCoolingLoops-1);

loopStart = 10000;
loopEnd = 100;
mul2 = nthroot(loopEnd/loopStart, numCoolingLoops-1);
numEquilbriumLoops = loopStart;

DeltaE_avg = 0.0;
counter = 0;

% Simulated Annealing Loop
for i = 1:numCoolingLoops
    disp(['Cycle: ', num2str(i), ' starting threshold: ', num2str(thCurrent*100), ' %']);
    
    numEquilbriumLoops = mul2 * numEquilbriumLoops;
    
    for j = 1:numEquilbriumLoops
        cityRoute_j = twoOptSwap(cityRoute_b); % Perturb the current route
        D_j = computeEUCDistance(numCities, cC, cityRoute_j);
        
        DeltaE = D_b - D_j;
        
        if DeltaE > 0 % Better solution
            accept = true;
        else % Worse solution
            if i == 1 && j == 1
                DeltaE_avg = abs(DeltaE); % Initialize DeltaE_avg
            end
            p = exp(-abs(DeltaE) / (DeltaE_avg * thCurrent^1.5)); % Acceptance probability
            accept = (rand() < p);
        end
        
        if accept
            cityRoute_b = cityRoute_j; % Accept the new route
            D_b = D_j; % Update best distance
            % Update DeltaE_avg
            numAcceptedSolutions = numAcceptedSolutions + 1.0;
            DeltaE_avg = (DeltaE_avg * (numAcceptedSolutions - 1.0) + abs(DeltaE)) / numAcceptedSolutions;
        end
    end
    
    % Restart mechanism: Reset every 500 cycles
    if mod(i, 500) == 0
        cityRoute_b = cityRoute_o;
        thCurrent = th1; % Reset the threshold (restart cooling)
    end
    
    % Update threshold
    thCurrent = mul1 * thCurrent; 
    
    % Check for convergence
    if abs(D_b - D_o) / D_o < 0.0001
        counter = counter + 1;
    end
    if counter == 100
        break;
    end
    
    Threshold(i) = thCurrent;
    cityRoute_o = cityRoute_b; % Update optimal route
    D(i+1) = D_b; % Record the route distance
    D_o = D_b; % Update optimal distance
end

% Print solution
fprintf("\n");
disp(['Best solution: ', num2str(cityRoute_o)]);
fprintf("\n");

% Compute final distance
D_b = computeEUCDistance(numCities, cC, cityRoute_o);

fprintf("Best algo   objective: %10.6f\n", D_b);
fprintf("Best global objective: %10.6f\n", D_o);

% Save best city route to file
fileID = fopen('BestCR.txt', 'w');
fprintf(fileID, '%6.2f\n', cityRoute_o);
fclose(fileID);

% Plot threshold
figure;
plot(1:numCoolingLoops, Threshold, 'LineWidth', 2);
ylabel('Threshold of acceptance', 'fontsize', 14, 'fontname', 'Arial');
xlabel('Route Number', 'fontsize', 14, 'fontname', 'Arial');
title('Threshold of acceptance vs Route Number', 'fontsize', 16, 'fontname', 'Arial'); 

% Plot distance vs route number
figure;
plot(D, 'r.-');
ylabel('Distance', 'fontsize', 14, 'fontname', 'Arial');
xlabel('Route Number', 'fontsize', 14, 'fontname', 'Arial');
title('Distance vs Route Number', 'fontsize', 16, 'fontname', 'Arial'); 

% Plot best route
L = zeros(numCities, 1);
for i = 1:numCities
    L(i) = cC(cityRoute_o(i), 1);
    x(i) = cC(cityRoute_o(i), 2);
    y(i) = cC(cityRoute_o(i), 3);
end
x(numCities+1) = cC(cityRoute_o(1), 2);
y(numCities+1) = cC(cityRoute_o(1), 3);

figure;
hold on;
plot(x', y', 'r', 'LineWidth', 1, 'MarkerSize', 8, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', [1.0, 1.0, 1.0]);
plot(x(1), y(1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', [1.0, 0.0, 0.0]);
labels = cellstr(num2str(L));
text(x(1:numCities)', y(1:numCities)', labels, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'center');
ylabel('Y Coordinate', 'fontsize', 18, 'fontname', 'Arial');
xlabel('X Coordinate', 'fontsize', 18, 'fontname', 'Arial');
title('Best City Route', 'fontsize', 20, 'fontname', 'Arial');

endTime = toc;
fprintf('Total time: %d minutes and %.1f seconds\n', floor(endTime/60), rem(endTime,60));

% Helper Functions
function newRoute = twoOptSwap(route)
    n = length(route);
    i = randi(n - 1);
    j = randi([i + 1, n]);
    newRoute = route;
    newRoute(i:j) = route(j:-1:i);
end

function route = nearestNeighbor(cC, numCities)
    visited = false(numCities, 1);
    route = zeros(1, numCities);
    route(1) = 1;
    visited(1) = true;
    for i = 2:numCities
        lastCity = route(i - 1);
        distances = pdist2(cC(lastCity, 2:3), cC(:, 2:3));
        distances(visited) = inf;
        [~, nextCity] = min(distances);
        route(i) = nextCity;
        visited(nextCity) = true;
    end
end

function distance = computeEUCDistance(numCities, cC, route)
    distance = 0;
    for i = 1:numCities - 1
        distance = distance + sqrt((cC(route(i), 2) - cC(route(i+1), 2))^2 + ...
                                   (cC(route(i), 3) - cC(route(i+1), 3))^2);
    end
    distance = distance + sqrt((cC(route(numCities), 2) - cC(route(1), 2))^2 + ...
                               (cC(route(numCities), 3) - cC(route(1), 3))^2);
end
