%% Phase 5 - Potential Flow over Symmetric Airfoil (Clean Version)
% This script computes inviscid potential flow around a symmetric airfoil.
% It enforces the no-penetration boundary condition by placing point sources
% on multiple symmetric rows near the chord line, then solving for source
% strengths using a weighted + regularized least-squares method.
% It also performs:
%   (1) Independence study vs number of sources N
%   (2) Thickness scaling study
%   (3) Field visualization of streamfunction psi and potential phi

clearvars -except data; clc; close all;  % Keep 'data' (airfoil points) but clear everything else; reset console; close figures

%% Settings
Uinf = 1.0;                 % Free-stream speed magnitude
AoA  = 0.0;                 % Angle of attack in degrees (converted later to radians)

Ntarget = [60 120 240 480]; % Target total numbers of chord singularities (will be adjusted to fit symmetric rows)
tScaleList = [0.5 1.0 1.5 2.0];          % Thickness scaling factors (scale y-coordinates of airfoil)
Nref_forThickness_target = 360;          % Reference N target used during thickness sweep

xlimField = [-0.5 1.5];     % x-limits for field plots (phi/psi contours)
ylimField = [-0.6 0.6];     % y-limits for field plots (phi/psi contours)
NxField = 260; NyField = 220; % Grid resolution for field evaluation

K = 3;                      % Number of positive-y source rows (total rows = 2*K due to symmetry)
sourceXPad = 0.02;          % Padding from leading/trailing edges for source placement along chord (avoid singularities at ends)
epsYOffsetMin  = 1e-4;      % Minimum offset away from y=0 for source rows (avoid exactly on chord line)

core2_base = 1e-6;          % Base "core radius^2" used to desingularize 1/r^2 behavior of point sources
regRel_base = 1e-7;         % Base relative regularization strength (ridge) for least-squares stability
useWeights = true;          % Use weighted boundary residuals (e.g., emphasize leading edge)

VnTrust = 1e-2;             % Trust threshold: if max|Vn| > this, computed forces may be unreliable

%% Input
% Expect 'data' in the MATLAB workspace: numeric matrix with at least [x y] columns
assert(exist('data','var')==1 && isnumeric(data) && size(data,2)>=2, ...
    'Expected numeric matrix "data" in workspace with columns at least [x y].');
xy_raw = data(:,1:2);       % Extract only x,y columns

%% Geometry + panels
% Clean, reorder, normalize, and orient the airfoil geometry:
% - Remove invalid points and near-duplicates
% - Ensure chord is aligned with x-axis, LE at x=0, TE at x=1
% - Ensure polygon is closed and CCW (for consistent outward normals)
[xy, ~] = preprocessAirfoilGeometry(xy_raw);

% Build panel representation from polyline points:
% - panel centers (xc,yc), lengths ds
% - tangents (tx,ty) and outward normals (nx,ny)
pan = buildPanelsFromPolyline(xy);

% Convert AoA to radians and define uniform free-stream velocity vector
alpha = deg2rad(AoA);
Vinf = [Uinf*cos(alpha), Uinf*sin(alpha)];

% Determine a representative half-thickness from panel center y-values
% Used to place source rows within the airfoil thickness band (but away from y=0)
tHalf = max(abs(pan.yc));
tHalf = max(tHalf, epsYOffsetMin);

% Choose K positive y-positions between 15% and 85% of half-thickness
% (then mirrored to negative y inside makeChordSourcesMultiRowSym)
yPos = linspace(0.15*tHalf, 0.85*tHalf, K).';
yPos = max(yPos, epsYOffsetMin);

%% Compatible N list
% Ensure total N is compatible with symmetric 2*K rows
Nlist = zeros(size(Ntarget));
for i = 1:numel(Ntarget)
    Nlist(i) = makeCompatibleTotalN(Ntarget(i), 2*K);
end

%% Figure (1): Geometry + chord + singularities (largest N)
% Create the chord source locations for the largest N (for visualization)
[xSrc_plot, ySrc_plot, ~] = makeChordSourcesMultiRowSym(Nlist(end), K, sourceXPad, yPos);

figure('Name','(1) Geometry + sources','Color','w');
plot(xy(:,1), xy(:,2), 'k-', 'LineWidth', 1.5); hold on; grid on; axis equal; % Airfoil polygon
plot([0 1],[0 0],'b--','LineWidth',1.2);                                      % Chord line
plot(xSrc_plot, ySrc_plot, 'ro', 'MarkerSize', 3, 'MarkerFaceColor','r');     % Source points
xlabel('x/c'); ylabel('y/c');
title('(1) Airfoil geometry, chord, and chord singularities');
legend('Airfoil boundary','Chord line','Chord singularities','Location','Best');

%% Independence study
% Study how the no-penetration error and force coefficients change with N
results = struct('N',[],'maxAbsVn',[],'rmsVn',[],'Cd',[],'CdAbs',[],'Cl',[]);

for kN = 1:numel(Nlist)
    N = Nlist(kN);

    % Build symmetric multi-row chord source locations
    % NxPerRow is determined internally based on N and number of rows (2*K)
    [xSrc, ySrc, NxPerRow] = makeChordSourcesMultiRowSym(N, K, sourceXPad, yPos);

    % Solver options: desingularization core and ridge regularization strength
    opts.core2 = core2_base;
    opts.regRel = regRel_base;

    % Solve for source strengths q using adaptive stabilization:
    % if the system is ill-conditioned or Cd is unreasonable, increase regularization/core
    [sol, ~] = solveChordSources_MultiRowSym_LS_adaptive( ...
        pan, xSrc, ySrc, K, NxPerRow, Vinf, opts, useWeights);

    % Store key metrics:
    % - Vn should be ~0 on the boundary (no-penetration)
    % - Cd should be ~0 in ideal inviscid potential flow (D'Alembert paradox), so |Cd| is a good error indicator
    results(kN).N = numel(xSrc);
    results(kN).maxAbsVn = max(abs(sol.Vn));
    results(kN).rmsVn = sqrt(mean(sol.Vn.^2));
    results(kN).Cd = sol.Cd;
    results(kN).CdAbs = abs(sol.Cd);
    results(kN).Cl = sol.Cl;
end

% Minimal console output for report
fprintf('Independence study (N, max|Vn|, |Cd|, Cl):\n');
for kN = 1:numel(results)
    fprintf('N=%4d  max|Vn|=%.3e  |Cd|=%.3e  Cl=%+.3e\n', ...
        results(kN).N, results(kN).maxAbsVn, results(kN).CdAbs, results(kN).Cl);
end
fprintf('\n');

%% Figure (2): independence plots
% Plot boundary normal velocity error and |Cd| versus N (log-scaled x-axis)
figure('Name','(2) Independence Study','Color','w');

subplot(1,2,1);
semilogx([results.N],[results.maxAbsVn],'o-','LineWidth',1.2); grid on;
yline(VnTrust,'k--','LineWidth',1.0); % Threshold line for "trustworthy" forces
xlabel('N'); ylabel('max(|V_n|)');
title('No-penetration error');
legend('max|Vn|',sprintf('threshold %g',VnTrust),'Location','Best');

subplot(1,2,2);
semilogx([results.N],[results.CdAbs],'s-','LineWidth',1.2); grid on;
xlabel('N'); ylabel('|C_d|');
title('|Cd| vs N');

%% Thickness study
% Investigate how thickness scaling affects boundary condition error and forces
Nref = makeCompatibleTotalN(Nref_forThickness_target, 2*K);

CdAbs_vs_t = zeros(size(tScaleList));
Cl_vs_t = zeros(size(tScaleList));
maxVn_vs_t = zeros(size(tScaleList));

for i = 1:numel(tScaleList)
    tScale = tScaleList(i);

    % Scale airfoil thickness by scaling y-coordinate
    xy_t = xy; xy_t(:,2) = tScale * xy_t(:,2);

    % Rebuild panels for the modified geometry
    pan_t = buildPanelsFromPolyline(xy_t);

    % Update thickness-based source-row y-locations accordingly
    tHalf_t = max(abs(pan_t.yc));
    tHalf_t = max(tHalf_t, epsYOffsetMin);
    yPos_t = linspace(0.15*tHalf_t, 0.85*tHalf_t, K).';
    yPos_t = max(yPos_t, epsYOffsetMin);

    % Build source locations for the reference N
    [xSrc, ySrc, NxPerRow] = makeChordSourcesMultiRowSym(Nref, K, sourceXPad, yPos_t);

    % Same base stabilization options
    opts.core2 = core2_base;
    opts.regRel = regRel_base;

    % Solve for source strengths for this thickness
    [sol_t, ~] = solveChordSources_MultiRowSym_LS_adaptive( ...
        pan_t, xSrc, ySrc, K, NxPerRow, Vinf, opts, useWeights);

    % Record metrics vs thickness
    maxVn_vs_t(i) = max(abs(sol_t.Vn));
    CdAbs_vs_t(i) = abs(sol_t.Cd);
    Cl_vs_t(i) = sol_t.Cl;
end

fprintf('Thickness study (tScale, max|Vn|, |Cd|, Cl):\n');
for i = 1:numel(tScaleList)
    fprintf('t=%.2f  max|Vn|=%.3e  |Cd|=%.3e  Cl=%+.3e\n', ...
        tScaleList(i), maxVn_vs_t(i), CdAbs_vs_t(i), Cl_vs_t(i));
end
fprintf('\n');

%% Figure (3): |Cd| vs thickness
% Visualize how the (ideally near-zero) drag error changes with thickness scaling
figure('Name','(3) Cd vs thickness scale','Color','w');
plot(tScaleList, CdAbs_vs_t, 'o-','LineWidth',1.5); grid on;
xlabel('Thickness scale'); ylabel('|C_d|');
title('(3) |Cd| vs thickness scale');

%% Pick best N = largest N
% For field visualization, we choose the largest N (most resolved)
[~, idxBest] = max([results.N]);
Nfield = results(idxBest).N;

% Build sources and solve once more for the chosen N
[xSrc, ySrc, NxPerRow] = makeChordSourcesMultiRowSym(Nfield, K, sourceXPad, yPos);
opts.core2 = core2_base; opts.regRel = regRel_base;
[solField, ~] = solveChordSources_MultiRowSym_LS_adaptive( ...
    pan, xSrc, ySrc, K, NxPerRow, Vinf, opts, useWeights);

%% Field grid + phi/psi
% Create grid for field evaluation in a larger box around the airfoil
xv = linspace(xlimField(1), xlimField(2), NxField);
yv = linspace(ylimField(1), ylimField(2), NyField);
[X,Y] = meshgrid(xv,yv);

% Evaluate velocity potential phi and streamfunction psi:
% phi = uniform-flow potential + sum(source potentials)
% psi = uniform-flow streamfunction + sum(source streamfunctions)
[phi, psi] = evalFieldPhiPsi(X,Y,xSrc,ySrc,solField.q,Vinf,solField.core2Used);

% Mask the interior of the airfoil polygon for plotting (set to NaN)
in = inpolygon(X, Y, xy(:,1), xy(:,2));
phi(in) = NaN; psi(in) = NaN;

% Find the x-location of maximum thickness (max |y|)
[~, imax] = max(abs(xy(:,2)));
x_thick = xy(imax,1);

%% Figure (4): psi contours
% Streamfunction contours correspond to streamlines in potential flow
figure('Name','(4) Streamfunction psi','Color','w');
contour(X,Y,psi,60); hold on; grid on; axis equal;
plot(xy(:,1), xy(:,2), 'k-','LineWidth',1.4);
xline(x_thick,'r--','LineWidth',1.8);
xlabel('x/c'); ylabel('y/c');
title('(4) Stream function \psi');
legend('\psi contours','Airfoil','Max thickness x','Location','Best');

%% Figure (5): phi contours
% Potential contours are equipotential lines
figure('Name','(5) Potential phi','Color','w');
contour(X,Y,phi,60); hold on; grid on; axis equal;
plot(xy(:,1), xy(:,2), 'k-','LineWidth',1.4);
xline(x_thick,'r--','LineWidth',1.8);
xlabel('x/c'); ylabel('y/c');
title('(5) Velocity potential \phi');
legend('\phi contours','Airfoil','Max thickness x','Location','Best');

%% -------------------- Helper functions --------------------
function Ncompat = makeCompatibleTotalN(Ntarget, nRows)
    % Make N compatible with the symmetric row structure: N must be a multiple of nRows
    Ncompat = nRows * ceil(Ntarget / nRows);
end

function [xy, info] = preprocessAirfoilGeometry(xy_in)
    % Remove rows with NaN/Inf
    xy_in = xy_in(~any(~isfinite(xy_in),2),:);

    % Remove near-duplicate points by rounding to tolerance and applying unique
    tol = 1e-10;
    xy = unique(round(xy_in/tol)*tol, 'rows', 'stable');

    % If first and last point are (nearly) identical, drop the last one (will re-close later)
    if norm(xy(1,:) - xy(end,:)) < 1e-12
        xy(end,:) = [];
    end

    % Basic safety: need enough points to define an airfoil boundary
    if size(xy,1) < 10, error('Not enough points after cleanup.'); end

    % Attempt to detect and fix bad point ordering (if polygon "jumps around"):
    % Use angles around centroid to reorder if needed.
    ctr = mean(xy,1);
    ang = unwrap(atan2(xy(:,2)-ctr(2), xy(:,1)-ctr(1)));
    if sum(diff(ang) < -pi/2) > 3
        ang2 = atan2(xy(:,2)-ctr(2), xy(:,1)-ctr(1));
        [~, idx] = sort(ang2);
        xy = xy(idx,:);
    end

    % Identify leading edge (min x) and trailing edge (max x)
    [~, iLE] = min(xy(:,1));
    [~, iTE] = max(xy(:,1));
    LE = xy(iLE,:);
    TE = xy(iTE,:);

    % Compute chord vector and chord length
    chordVec = TE - LE;
    c = norm(chordVec);
    if c < 1e-12, error('Chord too small.'); end

    % Rotate geometry so chord is aligned with +x direction
    theta = atan2(chordVec(2), chordVec(1));
    R = [cos(-theta) -sin(-theta); sin(-theta) cos(-theta)];

    % Translate so LE is at origin, then rotate, then scale by chord length
    xy = (R*(xy - LE).').';
    xy = xy / c;

    % Shift x so minimum x is at 0 (LE at x=0), then rescale so max x is 1 (TE at x=1)
    xy(:,1) = xy(:,1) - min(xy(:,1));
    xmax2 = max(xy(:,1));
    if xmax2 > 0, xy = xy / xmax2; end

    % Ensure polygon is explicitly closed (last point equals first point)
    if norm(xy(1,:) - xy(end,:)) > 1e-12
        xy = [xy; xy(1,:)];
    end

    % Ensure counter-clockwise orientation so outward normal formula is consistent
    if polygonSignedArea(xy) < 0
        xy = flipud(xy);
    end

    % Output some normalization info (can be useful for debugging)
    info.c = c;
    info.rotDeg = rad2deg(theta);
    info.LE = [0 0];
    info.TE = [1 0];
end

function A = polygonSignedArea(xy)
    % Signed area of a closed polygon: positive => CCW orientation
    x = xy(:,1); y = xy(:,2);
    A = 0.5*sum(x(1:end-1).*y(2:end) - x(2:end).*y(1:end-1));
end

function pan = buildPanelsFromPolyline(xy)
    % Make sure the polyline is closed
    if norm(xy(1,:) - xy(end,:)) > 1e-12
        xy = [xy; xy(1,:)];
    end

    % Panel endpoints
    x1 = xy(1:end-1,1); y1 = xy(1:end-1,2);
    x2 = xy(2:end,1);   y2 = xy(2:end,2);

    % Panel vectors and lengths
    dx = x2 - x1; dy = y2 - y1;
    ds = hypot(dx,dy);

    % Panel centers
    pan.xc = 0.5*(x1+x2);
    pan.yc = 0.5*(y1+y2);
    pan.ds = ds;

    % Unit tangents along panels
    tx = dx./ds; ty = dy./ds;
    pan.tx = tx; pan.ty = ty;

    % Unit outward normals for CCW polygons:
    % tangent t=[tx,ty], outward normal n=[ty,-tx]
    pan.nx = ty;        % outward for CCW
    pan.ny = -tx;
end

function [xSrc, ySrc, NxPerRow] = makeChordSourcesMultiRowSym(Ntotal, K, xPad, yPos)
    % Create symmetric chord source points arranged in 2*K rows:
    % +yPos(r) and -yPos(r) for r=1..K, with NxPerRow sources per row.

    nRows = 2*K;

    % Determine how many sources per row (at least 6), then adjust total to fit exactly
    NxPerRow = max(6, floor(Ntotal / nRows));
    NtotalAdj = NxPerRow * nRows;

    % X locations: evenly spaced along chord but padded away from x=0 and x=1
    xBase = linspace(xPad, 1-xPad, NxPerRow).';

    % Allocate arrays for all sources
    xSrc = zeros(NtotalAdj,1);
    ySrc = zeros(NtotalAdj,1);

    % Fill sources row by row (positive y then negative y for symmetry)
    idx = 1;
    for r = 1:K
        xSrc(idx:idx+NxPerRow-1) = xBase;
        ySrc(idx:idx+NxPerRow-1) = +yPos(r);
        idx = idx + NxPerRow;

        xSrc(idx:idx+NxPerRow-1) = xBase;
        ySrc(idx:idx+NxPerRow-1) = -yPos(r);
        idx = idx + NxPerRow;
    end
end

function [sol, diag] = solveChordSources_MultiRowSym_LS_adaptive(pan, xSrc, ySrc, K, NxPerRow, Vinf, opts, useWeights)
    % Adaptive wrapper: try solving with given (core2, regRel),
    % and increase them if the system appears ill-conditioned or produces non-physical results.

    core2 = opts.core2;
    regRel = opts.regRel;

    for attempt = 1:4
        [sol, diag] = solveChordSources_MultiRowSym_LS(pan, xSrc, ySrc, K, NxPerRow, Vinf, core2, regRel, useWeights);

        % If condition number is huge OR Cd is NaN/Inf OR Cd is absurdly large, increase stabilization
        if diag.condEst > 1e10 || ~isfinite(sol.Cd) || abs(sol.Cd) > 10
            regRel = regRel * 10;
            core2 = core2 * 10;
        else
            break;
        end
    end

    % Report the final core used (useful for field evaluation too)
    sol.core2Used = core2;
end

function [sol, diag] = solveChordSources_MultiRowSym_LS(pan, xSrc, ySrc, K, NxPerRow, Vinf, core2, regRel, useWeights)
    % Core least-squares solver for symmetric multi-row chord sources.
    %
    % Unknowns are source strengths per "unique" row (only K rows in +y),
    % and symmetry is enforced by mirroring those strengths to -y rows.

    M = numel(pan.xc);                   % Number of boundary panels (collocation points)
    Uref = hypot(Vinf(1),Vinf(2));       % Reference speed for Cp normalization

    % Optional weighting: emphasize leading edge region to better satisfy Vn there
    if useWeights
        w = 1 + 2*exp(-(pan.xc/0.12).^2); % Higher weight near small x (near LE)
        W = spdiags(sqrt(w), 0, M, M);    % Use sqrt(w) so W*(Aq-b) corresponds to weighted least squares
    else
        W = speye(M);
    end

    Nu = K * NxPerRow;                   % Number of unique unknowns (only +y rows)
    Ar = zeros(M, Nu);                   % Influence matrix mapping unknown strengths -> boundary normal velocity

    % Build Ar by summing the influence of symmetric +y and -y sources for each x-position in each row
    idxU = 1;
    idxFull = 1;
    for r = 1:K
        iPlus  = idxFull : idxFull+NxPerRow-1;              % indices of +y row sources for this r
        iMinus = idxFull+NxPerRow : idxFull+2*NxPerRow-1;   % indices of -y row sources for this r

        for j = 1:NxPerRow
            % Velocity induced at all panel centers by a unit-strength source at (+y row)
            [uxp, uyp] = velFromPointSource(pan.xc, pan.yc, xSrc(iPlus(j)),  ySrc(iPlus(j)),  1.0, core2);

            % Velocity induced at all panel centers by a unit-strength source at (-y row)
            [uxm, uym] = velFromPointSource(pan.xc, pan.yc, xSrc(iMinus(j)), ySrc(iMinus(j)), 1.0, core2);

            % Normal component from the pair (+y and -y), for this one unknown amplitude
            Ar(:,idxU) = (uxp+uxm).*pan.nx + (uyp+uym).*pan.ny;
            idxU = idxU + 1;
        end

        idxFull = idxFull + 2*NxPerRow;  % move to next pair of rows
    end

    % Right-hand side enforces no-penetration: (Vinf + Vinduced)·n = 0 -> Vinduced·n = -Vinf·n
    b = -(Vinf(1)*pan.nx + Vinf(2)*pan.ny);

    % Apply weighting (if used)
    Arw = W*Ar;
    bw  = W*b;

    % Enforce sum of unique strengths = 0 using a null-space parameterization:
    % a = Z*alpha => sum(a)=0 by construction.
    Z = [eye(Nu-1); -ones(1,Nu-1)];
    Arr = Arw * Z;

    % Estimate conditioning using singular values (rough condition estimate)
    s = svd(Arr,'econ');
    condEst = s(1)/max(s(end), eps);

    % Ridge regularization parameter (scaled relative to largest singular value)
    lambda = regRel * (s(1)^2 + eps);

    % Solve regularized least squares for alpha
    alpha = (Arr.'*Arr + lambda*eye(Nu-1)) \ (Arr.'*bw);

    % Recover unique strengths with zero-sum constraint
    a = Z*alpha;

    % Expand unique strengths a to full strengths q for 2*K rows:
    % Each +y row and its -y mirror share the same strength distribution.
    q = zeros(2*K*NxPerRow,1);
    idxU = 1; idxQ = 1;
    for r = 1:K
        arow = a(idxU:idxU+NxPerRow-1);
        q(idxQ:idxQ+NxPerRow-1) = arow;                      % +y row
        q(idxQ+NxPerRow:idxQ+2*NxPerRow-1) = arow;           % -y row (same strengths)
        idxU = idxU + NxPerRow;
        idxQ = idxQ + 2*NxPerRow;
    end

    % Compute total boundary velocity = free-stream + contributions from all sources
    Vx = Vinf(1)*ones(M,1);
    Vy = Vinf(2)*ones(M,1);
    for j = 1:numel(q)
        [ux, uy] = velFromPointSource(pan.xc, pan.yc, xSrc(j), ySrc(j), q(j), core2);
        Vx = Vx + ux; Vy = Vy + uy;
    end

    % Boundary normal velocity (should be near zero)
    Vn = Vx.*pan.nx + Vy.*pan.ny;

    % Speed magnitude and pressure coefficient via Bernoulli (inviscid, steady, irrotational)
    Vmag = hypot(Vx,Vy);
    Cp = 1 - (Vmag./Uref).^2;

    % Integrate pressure forces over the surface:
    % Force per panel ~ -Cp * n * ds (negative because pressure acts inward)
    Fx = -sum(Cp .* pan.nx .* pan.ds);
    Fy = -sum(Cp .* pan.ny .* pan.ds);

    % Pack solution
    sol.q = q; sol.Vn = Vn; sol.Cp = Cp;
    sol.Cd = Fx; sol.Cl = Fy;   % Here Cd, Cl are treated as nondimensional-like (chord normalized to 1)

    % Diagnostics for stability/conditioning
    diag.condEst = condEst;
    diag.lambda = lambda;
    diag.core2 = core2;
end

function [ux, uy] = velFromPointSource(x, y, xs, ys, q, core2)
    % Velocity induced by a 2D point source of strength q located at (xs,ys).
    % For a source: u = (q/(2*pi)) * r / r^2, with r = [x-xs, y-ys].
    % core2 prevents singularity by enforcing r^2 >= core2.
    dx = x - xs; dy = y - ys;
    r2 = max(dx.^2 + dy.^2, core2);
    coef = (q/(2*pi)) ./ r2;
    ux = coef .* dx;
    uy = coef .* dy;
end

function [phi, psi] = evalFieldPhiPsi(X,Y,xSrc,ySrc,q,Vinf,core2)
    % Evaluate velocity potential phi and streamfunction psi on a grid (X,Y).
    % Uniform flow contributions:
    %   phi_uniform = Vinf_x * X + Vinf_y * Y
    %   psi_uniform = Vinf_x * Y - Vinf_y * X
    %
    % Source contributions:
    %   phi_source = (q/(2*pi)) * log(r)
    %   psi_source = (q/(2*pi)) * atan2(dy, dx)
    phi = Vinf(1)*X + Vinf(2)*Y;
    psi = Vinf(1)*Y - Vinf(2)*X;

    for j = 1:numel(q)
        dx = X - xSrc(j); dy = Y - ySrc(j);
        r2 = max(dx.^2 + dy.^2, core2);
        r = sqrt(r2);
        phi = phi + (q(j)/(2*pi))*log(r);
        psi = psi + (q(j)/(2*pi))*atan2(dy, dx);
    end
end