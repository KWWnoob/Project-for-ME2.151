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

% Damping Matrix (C)
Damping = zeros(3, 3);

% Stiffness Matrix (K)
K = diag([k, k, k]);

F_matrix_4_tendons = [-1 -1 +1 +1;
            -1 +1 -1 +1;
            -1 +1 +1 -1];

F_matrix_1_tendon = [1;1;1];

% Define LQR weights
Q = diag([10, 10, 10, 1, 1, 1]); % State cost
R_1_tendon = 0.00001;                  % Control cost for single tendon
R_4_tendons = eye(4)*0.000001;            % Control cost for four tendons

A = [zeros(3) eye(3);
    -inv(M)*K zeros(3)];

% LQR for B_1_tendon
A_1_tendon = A; % Use the same A matrix
B_1_tendon = [zeros(3, 1); F_matrix_1_tendon];
K_1_tendon = lqr(A_1_tendon, B_1_tendon, Q, R_1_tendon);

% LQR for B_4_tendons
B_4_tendons = [zeros(3, 4); F_matrix_4_tendons];
K_4_tendons = lqr(A, B_4_tendons, Q, R_4_tendons);

% Display the LQR gain matrices
disp('LQR Gain Matrix for 1 Tendon (K_1_tendon):');
disp(K_1_tendon);
disp('LQR Gain Matrix for 4 Tendons (K_4_tendons):');
disp(K_4_tendons);

C = [0 0 1 0 0 0]; D = 0;
sys = ss(A,B_1_tendon,C,D);
% pole placement

p = [-3+3i -3-3i -5+3i -5+3i -10 -15];
K_placement_1_tendon = place(A, B_1_tendon, p);

sys1 = ss(A - B_1_tendon*K_1_tendon, B_1_tendon, C, D);
sys2 = ss(A - B_4_tendons*K_4_tendons, B_4_tendons, C, D);
