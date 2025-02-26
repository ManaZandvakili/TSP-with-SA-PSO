clc; clear; close all; tic;

% Load city coordinates
cC = load('Desktop\dj44.txt');
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

% Parameters
numCoolingLoops = 500;
th1 = 0.3; % Initial threshold
th2 = 0.00001; % Final threshold
mul1 = nthroot(th2/th1, numCoolingLoops-1); % Cooling rate
loopStart = 10000; % Initial number of equilibrium iterations
loopEnd = 100; % Final number of equilibrium iterations
beta = nthroot(loopEnd/loopStart, numCoolingLoops-1);

% Define the number of candidates
numCandidates = 4; % Number of parallel simulated annealing instances

% Create storage for candidate routes and distances
cityRoutes = cell(1, numCandidates); % Stores the best route for each candidate
bestDistances = zeros(1, numCandidates); % Stores the best distance for each candidate

% Parallel processing setup
parfor candidateIdx = 1:numCandidates
    % Local variables for each candidate
    localCityRoute = nearestNeighbor(cC, numCities); % Initial route
    localCityRoute_b = localCityRoute;
    localCityRoute_o = localCityRoute;
    localD_b = computeEUCDistance(numCities, cC, localCityRoute);
    localD_o = localD_b;
    localThreshold = th1;
    localDeltaE_avg = 0.0;
    localNumAcceptedSolutions = 1.0;
    localNumEquilbriumLoops = loopStart;

    % Simulated annealing for this candidate
    for i = 1:numCoolingLoops
        % Inner loop for thermal equilibrium
        for j = 1:localNumEquilbriumLoops
            cityRoute_j = twoOptSwap(localCityRoute_b); % Perturb the current route
            D_j = computeEUCDistance(numCities, cC, cityRoute_j);

            DeltaE = localD_b - D_j;

            % Acceptance logic
            if DeltaE > 0
                accept = true;
            else
                if i == 1 && j == 1
                    localDeltaE_avg = abs(DeltaE); % Initialize DeltaE_avg
                end
                p = exp(-abs(DeltaE) / (localDeltaE_avg * localThreshold^1.5)); % Acceptance probability
                accept = (rand() < p);
            end

            if accept
                localCityRoute_b = cityRoute_j;
                localD_b = D_j;
                localNumAcceptedSolutions = localNumAcceptedSolutions + 1.0;
                localDeltaE_avg = (localDeltaE_avg * (localNumAcceptedSolutions - 1.0) + abs(DeltaE)) / localNumAcceptedSolutions;
            end
        end

        % Update threshold
        localThreshold = mul1 * localThreshold;
        localCityRoute_o = localCityRoute_b;
        localD_o = localD_b;

        % Optional restart mechanism for this candidate
        if mod(i, 500) == 0
            localCityRoute_b = localCityRoute_o;
            localThreshold = th1; % Reset the threshold
        end
    end

    % Store results for this candidate
    cityRoutes{candidateIdx} = localCityRoute_o;
    bestDistances(candidateIdx) = localD_o;
end

% Find the global best solution across all candidates
[globalBestDistance, bestCandidateIdx] = min(bestDistances);
globalBestRoute = cityRoutes{bestCandidateIdx};

% Print the global best solution
fprintf("\n");
disp(['Best solution: ', num2str(globalBestRoute)]);
fprintf("Global best objective: %10.6f\n", globalBestDistance);

% Plot the best route
figure;
L = zeros(numCities, 1);
for i = 1:numCities
    L(i) = cC(globalBestRoute(i), 1);
    x(i) = cC(globalBestRoute(i), 2);
    y(i) = cC(globalBestRoute(i), 3);
end
x(numCities+1) = cC(globalBestRoute(1), 2);
y(numCities+1) = cC(globalBestRoute(1), 3);

plot(x', y', 'r', 'LineWidth', 1, 'MarkerSize', 8, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', [1.0, 1.0, 1.0]);
plot(x(1), y(1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', [1.0, 0.0, 0.0]);
labels = cellstr(num2str(L));
text(x(1:numCities)', y(1:numCities)', labels, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'center');
ylabel('Y Coordinate', 'fontsize', 18, 'fontname', 'Arial');
xlabel('X Coordinate', 'fontsize', 18, 'fontname', 'Arial');
title('Global Best City Route', 'fontsize', 20, 'fontname', 'Arial');
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
