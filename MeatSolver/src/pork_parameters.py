U_INFTY = 0.1  # m/s Maximal flame speed is about 1 m/s for propane. Fudge factor for diffusion
PR = 0.7
MU = 2.2691E-5  # Pa.s @ T_film=250 F http://www.mhtl.uwaterloo.ca/old/onlinetools/airprop/airprop.html
D = 5e-2  # m
DENSITY = 0.89500  # kg/m^3 @ T_film=250 F, 1 atm http://www.mhtl.uwaterloo.ca/old/onlinetools/airprop/airprop.html
RE = DENSITY * U_INFTY * D / MU
NUM_MODES = 20
NUM_TAU = 3
NUM_RHO = 3
NUM_THETA0 = 3
NUM_KAPPA = 3

NUD = 10  # for Re~=200 and Pr=0.7 for mean flow around cylinder (p. 427 of Incopera)

RHO = 1090  # [kg/m^3]
K = 0.47  # [W/m.K] src: https://www.engineeringtoolbox.com/food-thermal-conductivity-d_2177.html
C = 2760  # [J/kg.K] src: https://www.engineeringtoolbox.com/specific-heat-capacity-food-d_295.html
ALPHA = K / (RHO * C)
MAX_COOK_TIME = 3600  # [s]
MIN_COOK_TIME = 60  # [s]
MAX_D = 12 * 0.0254  # [m]
MIN_D = 0.0254  # [m]
MAX_KAPPA = ALPHA * MAX_COOK_TIME / MIN_D**2.0
MIN_KAPPA = ALPHA * MIN_COOK_TIME / MAX_D**2.0