clc; clear; close all; tic;

% Load city coordinates
cC = load('Desktop\dj44.txt');
numCities = size(cC, 1);
x = cC(1:numCities, 2);
y = cC(1:numCities, 3);
x(numCities+1) = cC(1, 2); % Connect the last city to the first
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

% PSO Parameters
numParticles = 30; % Number of particles in the swarm
maxIterations = 50; % Maximum number of PSO iterations
w = 0.8; % Inertia weight
c1 = 1.5; % Cognitive coefficient
c2 = 1.5; % Social coefficient

% SA Parameters
numCoolingLoops = 100; % Number of cooling cycles for SA
numEquilbriumLoops = 1000; % Initial number of equilibrium iterations
th1 = 0.3; % Initial threshold
th2 = 0.00001; % Final threshold
mul1 = nthroot(th2/th1, numCoolingLoops-1);

% Initialize particles
particles = cell(numParticles, 1); % Each particle represents a route
localBest = cell(numParticles, 1); % Local best route for each particle
localBestDistance = inf(numParticles, 1); % Distance for each local best
globalBest = []; % Global best route
globalBestDistance = inf; % Global best distance

% Random initialization of particles
for p = 1:numParticles
    particles{p} = randperm(numCities); % Random route
    dist = computeEUCDistance(numCities, cC, particles{p});
    localBest{p} = particles{p};
    localBestDistance(p) = dist;
    if dist < globalBestDistance
        globalBest = particles{p};
        globalBestDistance = dist;
    end
end

% Hybrid PSO and SA Main Loop
for iter = 1:maxIterations
    for p = 1:numParticles
        % Evaluate fitness of current particle
        currentDistance = computeEUCDistance(numCities, cC, particles{p});
        if currentDistance < localBestDistance(p)
            localBest{p} = particles{p};
            localBestDistance(p) = currentDistance;
        end
        if currentDistance < globalBestDistance
            globalBest = particles{p};
            globalBestDistance = currentDistance;
        end
        
        % Refine particle using SA
        [refinedRoute, refinedDistance] = simulatedAnnealing(particles{p}, cC, numCoolingLoops, numEquilbriumLoops, th1, mul1);
        if refinedDistance < localBestDistance(p)
            localBest{p} = refinedRoute;
            localBestDistance(p) = refinedDistance;
        end
        if refinedDistance < globalBestDistance
            globalBest = refinedRoute;
            globalBestDistance = refinedDistance;
        end
        
        % Update particle position (route) using PSO logic
        particles{p} = updateParticle(particles{p}, localBest{p}, globalBest, w, c1, c2);
    end
    
    % Display progress
    if mod(iter, 50) == 0
        fprintf('Iteration %d: Best Distance = %.4f\n', iter, globalBestDistance);
    end
end

% Display final results
fprintf('Final Best Distance: %.4f\n', globalBestDistance);
disp(['Best Route: ', num2str(globalBest)]);

% Plot the best route
figure;
L = zeros(numCities, 1);
for i = 1:numCities
    L(i) = cC(globalBest(i), 1);
    x(i) = cC(globalBest(i), 2);
    y(i) = cC(globalBest(i), 3);
end
x(numCities+1) = cC(globalBest(1), 2);
y(numCities+1) = cC(globalBest(1), 3);

% Plot route between cities
plot(x', y', 'r', 'LineWidth', 1, 'MarkerSize', 8, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', [1.0, 1.0, 1.0]);
plot(x(1), y(1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', [1.0, 0.0, 0.0]);
labels = cellstr(num2str(L));
text(x(1:numCities)', y(1:numCities)', labels, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'center');
ylabel('Y Coordinate', 'fontsize', 18, 'fontname', 'Arial');
xlabel('X Coordinate', 'fontsize', 18, 'fontname', 'Arial');
title('Best City Route (Hybrid PSO and SA)', 'fontsize', 20, 'fontname', 'Arial');

% Save the best route to a text file
fileID = fopen('bestroute.txt', 'w');
fprintf(fileID, '%d\n', globalBest); % Write each city index in the best route
fclose(fileID);

endTime = toc;
fprintf('Total time: %d minutes and %.1f seconds\n', floor(endTime/60), rem(endTime,60));

% Helper Functions
function [refinedRoute, refinedDistance] = simulatedAnnealing(route, cC, numCoolingLoops, numEquilbriumLoops, th1, mul1)
    refinedRoute = route;
    refinedDistance = computeEUCDistance(length(route), cC, route);
    thCurrent = th1;
    for i = 1:numCoolingLoops
        for j = 1:numEquilbriumLoops
            newRoute = twoOptSwap(refinedRoute); % Call twoOptSwap here
            newDistance = computeEUCDistance(length(route), cC, newRoute);
            DeltaE = refinedDistance - newDistance;
            if DeltaE > 0 || rand() < exp(DeltaE / thCurrent)
                refinedRoute = newRoute;
                refinedDistance = newDistance;
            end
        end
        thCurrent = mul1 * thCurrent; % Update temperature
    end
end

function updatedParticle = updateParticle(particle, localBest, globalBest, w, c1, c2)
    n = length(particle);
    updatedParticle = particle;
    for i = 1:n
        if rand < w
            % Swap cities randomly based on inertia
            idx1 = randi(n);
            idx2 = randi(n);
            updatedParticle([idx1, idx2]) = updatedParticle([idx2, idx1]);
        end
        if rand < c1
            % Move towards local best
            idx = find(particle == localBest(i));
            updatedParticle([i, idx]) = updatedParticle([idx, i]);
        end
        if rand < c2
            % Move towards global best
            idx = find(particle == globalBest(i));
            updatedParticle([i, idx]) = updatedParticle([idx, i]);
        end
    end
end

function newRoute = twoOptSwap(route)
    n = length(route);
    i = randi(n - 1); % Random index from 1 to n-1
    j = randi([i + 1, n]); % Random index greater than i and up to n
    newRoute = route;
    newRoute(i:j) = route(j:-1:i); % Reverse the order of cities between i and j
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
