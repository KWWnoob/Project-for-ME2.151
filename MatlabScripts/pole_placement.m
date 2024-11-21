% Define constants
m1 = 1.0; m2 = 1.0; k = 10.0;       % Masses and spring constant
F1 = 1.0; F2 = 1.0; F3 = 1.0; F4 = 1.0; % External forces
l = 5; r = 2;                       % Length of rod and radius of disk

% Moments of inertia
I1 = 1 / 12 * m1 * l^2;
I2 = m2 * r^2 / 2;

% Mass Matrix (M)
M = [ ...
    I1 + I2 + 5*m1*(l + r)^2 + 8.75*m2*(l + r)^2, 4*m1*(l + r)^2 + 8.5*m2*(l + r)^2, 6.25*m2*(l + r)^2; ...
    4*m1*(l + r)^2 + 8.5*m2*(l + r)^2, I1 + I2 + 4*m1*(l + r)^2 + 8.5*m2*(l + r)^2, 6.25*m2*(l + r)^2; ...
    6.25*m2*(l + r)^2, 6.25*m2*(l + r)^2, I1 + I2 + 6.25*m2*(l + r)^2 ...
];

% Damping Matrix (C) - simple damping model
Damping = 0.1 * eye(3); % Add some damping for stability

% Stiffness Matrix (K)
K = diag([k, k, k]);

% Force matrices for tendon configurations
F_matrix_4_tendons = [-1 -1 +1 +1;
                      -1 +1 -1 +1;
                      -1 +1 +1 -1];

F_matrix_1_tendon = [1;1;1];

% System matrices
A = [zeros(3) eye(3);
     -inv(M)*K zeros(3)];
B_1_tendon = [zeros(3, 1); F_matrix_1_tendon];
B_4_tendons = [zeros(3, 4); F_matrix_4_tendons];

C = eye(6); % Now we output all state variables (q1, q2, q3)
D = 0;

% Pole placement for feedback control
p = [-3 -4 -5 -6 -10 -15];  % Desired poles
K_placement_1_tendon = place(A, B_1_tendon, p);
K_placement_4_tendon = place(A, B_4_tendons, p);

% State-space system with feedback control
sys1 = ss(A - B_1_tendon*K_placement_1_tendon, B_1_tendon, C, D);
sys2 = ss(A - B_4_tendons*K_placement_4_tendon, B_4_tendons, C, D);

% Define initial conditions for the simulation (initial positions q1, q2, q3, and velocities)
initial_conditions = [0.2; 0.2; 0.2; 0; 0; 0]; % initial state: [q1; q2; q3; q1_dot; q2_dot; q3_dot]

% Time vector for simulation
t = 0:0.01:10;  % Simulate for 10 seconds with 0.01s time step

% Simulate the system dynamics (numerical integration using lsim)
[y, ~] = initial(sys2,initial_conditions,t);

% Check the size of y
disp('Size of y:');
disp(size(y));

% Extract the positions (q1, q2, q3) from the output of the simulation
q1 = y(:, 1);
q2 = y(:, 2);
q3 = y(:, 3);

% Plot the results
figure;
subplot(3, 1, 1);
plot(t, q1);
title('Position q1');
xlabel('Time (s)');
ylabel('q1 (rad)');

subplot(3, 1, 2);
plot(t, q2);
title('Position q2');
xlabel('Time (s)');
ylabel('q2 (rad)');

subplot(3, 1, 3);
plot(t, q3);
title('Position q3');
xlabel('Time (s)');
ylabel('q3 (rad)');
