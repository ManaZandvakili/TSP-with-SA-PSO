% Traveling Salesman Problem with Binary Integer Programming 
filename = 'dj44.txt'; 
stopsData = load(filename); % Load the text file into a matrix
stopsLon = stopsData(:, 2); % Extract longitude (x-coordinates)
stopsLat = stopsData(:, 3); % Extract latitude (y-coordinates)
nStops = size(stopsData, 1); % Number of stops


load('usborder.mat', 'x', 'y', 'xx', 'yy');

% Generate all possible trips (all pairs of stops)
idxs = nchoosek(1:nStops, 2);

% Calculate distances for each trip
dist = hypot(stopsLat(idxs(:, 1)) - stopsLat(idxs(:, 2)), ...
             stopsLon(idxs(:, 1)) - stopsLon(idxs(:, 2)));
lendist = length(dist);

% Visualize and save the initial map
figure;
G_empty = graph([], [], [], nStops); % Create a graph with nodes but no edges
hGraph = plot(G_empty, 'XData', stopsLon, 'YData', stopsLat, ...
              'LineStyle', 'none', 'NodeLabel', {});
hold on;
plot(x, y, 'r-'); % Draw US border
hold off;
title('Initial Map with Stops');
saveas(gcf, 'Initial_Map.png');

% Constraints: Each stop must be visited exactly twice
Aeq = spalloc(nStops, size(idxs, 1), nStops * (nStops - 1));
for ii = 1:nStops
    whichIdxs = (idxs == ii);
    whichIdxs = sparse(sum(whichIdxs, 2));
    Aeq(ii, :) = whichIdxs';
end
beq = 2 * ones(nStops, 1);

% Define binary bounds for decision variables
intcon = 1:lendist;
lb = zeros(lendist, 1);
ub = ones(lendist, 1);

% Start timing the simulation
simulationStartTime = tic; % Start timer
disp('Starting the optimization to find an initial solution...');
opts = optimoptions('intlinprog', 'Display', 'off');
[x_tsp, costopt, exitflag, output] = intlinprog(dist, intcon, [], [], Aeq, beq, lb, ub, opts);

% Round and convert solution to logical
x_tsp = logical(round(x_tsp));

% Create graph with solution trips
Gsol = graph(idxs(x_tsp, 1), idxs(x_tsp, 2), [], nStops);

% Visualize and save the initial solution with subtours
figure;
hGraph = plot(Gsol, 'XData', stopsLon, 'YData', stopsLat, ...
              'LineStyle', '-');
hold on;
plot(x, y, 'r-'); % Draw US border
title('Initial Solution with Subtours');
saveas(gcf, 'Initial_Solution_With_Subtours.png');

% Eliminate subtours iteratively and save intermediate results
disp('Starting subtour elimination process...');
tourIdxs = conncomp(Gsol);
numtours = max(tourIdxs);
A = spalloc(0, lendist, 0);
b = [];
iteration = 1;

while numtours > 1
    fprintf('Iteration %d: Found %d subtours\n', iteration, numtours);
    
    % Add subtour elimination constraints
    for ii = 1:numtours
        subTourIdx = find(tourIdxs == ii);
        variations = nchoosek(1:length(subTourIdx), 2);
        newConstraint = sparse(1, lendist);
        for jj = 1:size(variations, 1)
            whichVar = (sum(idxs == subTourIdx(variations(jj, 1)), 2)) & ...
                       (sum(idxs == subTourIdx(variations(jj, 2)), 2));
            newConstraint = newConstraint + whichVar';
        end
        A = [A; newConstraint];
        b = [b; length(subTourIdx) - 1];
    end
    
    % Re-optimize with new constraints
    [x_tsp, costopt, exitflag, output] = intlinprog(dist, intcon, A, b, Aeq, beq, lb, ub, opts);
    x_tsp = logical(round(x_tsp));
    Gsol = graph(idxs(x_tsp, 1), idxs(x_tsp, 2), [], nStops);
    
    % Save the updated graph
    figure;
    hGraph = plot(Gsol, 'XData', stopsLon, 'YData', stopsLat, ...
                  'LineStyle', '-');
    hold on;
    plot(x, y, 'r-'); % Draw US border
    title(['Iteration ' num2str(iteration) ': Updated Solution']);
    saveas(gcf, ['Iteration_' num2str(iteration) '.png']);
    
    % Update for next iteration
    tourIdxs = conncomp(Gsol);
    numtours = max(tourIdxs);
    iteration = iteration + 1;
end

% Stop timing the simulation
simulationEndTime = toc(simulationStartTime);

% Visualize and save the final solution
figure;
hGraph = plot(Gsol, 'XData', stopsLon, 'YData', stopsLat, ...
              'LineStyle', '-');
hold on;
plot(x, y, 'r-'); % Draw US border
title('Final Solution Without Subtours');
saveas(gcf, 'Final_Solution.png');

% Calculate the total distance traveled
totalDistance = sum(dist(x_tsp));
fprintf('Total distance traveled by the salesperson: %.2f units\n', totalDistance);
fprintf('Total simulation time: %.2f seconds\n', simulationEndTime);
disp('Subtour elimination completed.');
