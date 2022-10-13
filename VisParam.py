import json
from datetime import datetime

import matplotlib.path as mpath
import numpy as np
from django.utils.dateparse import parse_duration
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class VisParam:
    """
    defines Parameters of the visualisation

        :parameter

        :returns
            visualisation parameters
    """

    # ranges of values of specific variables that get mapped to colormap
    variable_plotting_bounds = {
        'TQC': (0, 0.25),
        'TQV': (0, 50),
        'TQI': (0, 0.2),
        'TOT_PREC': (0, 5),
        'W_SNOW': (0.001, 0.01),
        'elev': (0, 6000),
        'ice': (0.9, 1),
        'LPI': (20, 1000),
        'DHAIL_MX': (0, 1000)
    }

    draw_grid_lines = False
    grid_labelsize = 15
    time_font_size = 13

    dpi = 'figure'  # dpi of the picture 'figure' is the dpi of the figure itself, but can be changed to any number
    use_native_resolution = True  # automatically set picture sizes to data size
    # if use_native_resolution == False use this:
    figsize = (3840 / 100, 2160 / 100)

    do_precipitation_colourbar = False # experimental, is a bit messy and not pretty

    # main land parameter to tweak in interval (0,1)
    saturation_percentage = 0.60


    # defining color maps
    # Land
    colors = ['#f0d8af', '#f0d8af', '#f0d8af', '#ecce9a', '#ecce9a', '#e4b96f', '#e4b96f', '#a98e5c', '#a98e5c',
              '#6e6349', '#607a30', '#607a30', '#276618', '#276618', '#083613', '#083613']

    cmap_tmp = LinearSegmentedColormap.from_list("land", colors)
    my_cmap = cmap_tmp(np.linspace(0.0, 1.0, cmap_tmp.N))
    my_cmap = ListedColormap(my_cmap)
    cmap_land = my_cmap

    cmap_tmp = plt.cm.Blues
    my_cmap = cmap_tmp(np.arange(cmap_tmp.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap_tmp.N)
    my_cmap = ListedColormap(my_cmap)
    cmap_ocean = my_cmap

    # Snow
    cmap_tmp = plt.cm.binary_r
    my_cmap = cmap_tmp(np.linspace(0.7, 1.0, cmap_tmp.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap_tmp.N)
    my_cmap = ListedColormap(my_cmap)
    cmap_snow = my_cmap

    # Elevation
    colors = ["#854830", "#854830", "#422f2a", "#855f55", "#90706d"]
    cmap_tmp = LinearSegmentedColormap.from_list("elev", colors)
    my_cmap = cmap_tmp(np.linspace(0.0, 1.0, cmap_tmp.N))
    x = np.linspace(0, 1, cmap_tmp.N)
    my_cmap[:, -1] = np.linspace(0., 1., len(x))
    my_cmap = ListedColormap(my_cmap)
    cmap_elev = my_cmap

    # ice
    cmap_tmp = plt.cm.binary_r
    my_cmap = cmap_tmp(np.linspace(0.7, 1.0, cmap_tmp.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap_tmp.N)
    my_cmap = ListedColormap(my_cmap)
    cmap_ice = my_cmap

    # TQC cmap_tmp
    cmap_tmp = plt.cm.binary_r
    my_cmap = cmap_tmp(np.linspace(0.7, 1.0, cmap_tmp.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap_tmp.N)
    my_cmap = ListedColormap(my_cmap)
    cmap_TQC = my_cmap

    # TQV cmap_tmp
    cmap_tmp = plt.cm.inferno_r
    my_cmap = cmap_tmp(np.linspace(0.0, 0.35, cmap_tmp.N))
    my_cmap[:, -1] = np.linspace(0.10, 0.28, cmap_tmp.N)
    my_cmap = ListedColormap(my_cmap)
    cmap_TQV = my_cmap

    # TQI cmap_tmp
    colors = ['#66f5fa', '#ffffff']  # 7
    cmap_tmp = LinearSegmentedColormap.from_list("tqi", colors)
    my_cmap = cmap_tmp(np.linspace(0.3, 1.0, cmap_tmp.N))
    my_cmap[:, -1] = np.linspace(0.0, 1.0, cmap_tmp.N) ** (1 / 2)
    my_cmap = ListedColormap(my_cmap)
    cmap_TQI = my_cmap

    # TOT_PREC cmap_tmp
    cmap_tmp = plt.cm.gnuplot2_r
    my_cmap = cmap_tmp(np.arange(cmap_tmp.N))
    my_cmap[:, -1] = np.linspace(0.5, 1.0, cmap_tmp.N)
    my_cmap[0, -1] = 0
    my_cmap = ListedColormap(my_cmap)
    cmap_TOT_PREC = my_cmap

    # QV2M cmap_tmp
    cmap_tmp = plt.cm.inferno
    my_cmap = cmap_tmp(np.linspace(0.0, 1.00, cmap_tmp.N))
    my_cmap = ListedColormap(my_cmap)
    cmap_QV2M = my_cmap

    # for markers uni-colored colormaps
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0, 1])
    newcolors[:, :] = black
    cmap_black = ListedColormap(newcolors)

    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    newcolors[:, :] = white
    cmap_white = ListedColormap(newcolors)

    # finer land parameters
    deepest_soil_level_grass = 0.8
    deepest_soil_level_forest = 6.0
    land_plant_max_growth_sat_soil = 0.6
    forest_plant_max_growth_sat_soil = 0.2
    min_forest_greenness = 0.4

    marker_lightning = [[0.2, 1], [-0.4, 0.005], [0.45, 0.005], [-0.2, -1], [0.4, -0.005], [-0.45, -0.005]]

    # custom hail marker created with the help of the following tutorial
    # https://petercbsmith.github.io/marker-tutorial.html
    marker_hail = mpath.Path(np.array([[9.09448110e+01, -1.31214158e+02],
                                       [8.01948110e+01, -1.34714158e+02],
                                       [7.84448110e+01, -1.35844158e+02],
                                       [7.21948110e+01, -1.42224158e+02],
                                       [5.79448110e+01, -1.57104158e+02],
                                       [5.64448110e+01, -1.61354158e+02],
                                       [5.74448110e+01, -1.84124158e+02],
                                       [5.79448110e+01, -1.95634158e+02],
                                       [5.80748110e+01, -1.96134158e+02],
                                       [6.34448110e+01, -2.03884158e+02],
                                       [7.15748110e+01, -2.15764158e+02],
                                       [7.30748110e+01, -2.17014158e+02],
                                       [8.46948110e+01, -2.22644158e+02],
                                       [9.46948110e+01, -2.27654158e+02],
                                       [9.59448110e+01, -2.27904158e+02],
                                       [1.06694811e+02, -2.28024158e+02],
                                       [1.17694811e+02, -2.28024158e+02],
                                       [1.18314811e+02, -2.27904158e+02],
                                       [1.29074811e+02, -2.22394158e+02],
                                       [1.38694811e+02, -2.17524158e+02],
                                       [1.41074811e+02, -2.15644158e+02],
                                       [1.46194811e+02, -2.08884158e+02],
                                       [1.53574811e+02, -1.99134158e+02],
                                       [1.55194811e+02, -1.93624158e+02],
                                       [1.55194811e+02, -1.77994158e+02],
                                       [1.55074811e+02, -1.63364158e+02],
                                       [1.53444811e+02, -1.57984158e+02],
                                       [1.45944811e+02, -1.48354158e+02],
                                       [1.39314811e+02, -1.39844158e+02],
                                       [1.33944811e+02, -1.35844158e+02],
                                       [1.24314811e+02, -1.31964158e+02],
                                       [1.11574811e+02, -1.26834158e+02],
                                       [1.04814811e+02, -1.26714158e+02],
                                       [9.09448110e+01, -1.31214158e+02],
                                       [9.09448110e+01, -1.31214158e+02],
                                       [-1.70925189e+02, -1.31714158e+02],
                                       [-1.79305189e+02, -1.35094158e+02],
                                       [-1.83685189e+02, -1.37714158e+02],
                                       [-1.88055189e+02, -1.42094158e+02],
                                       [-1.96055189e+02, -1.49854158e+02],
                                       [-2.01685189e+02, -1.63234158e+02],
                                       [-2.02555189e+02, -1.76114158e+02],
                                       [-2.03055189e+02, -1.84624158e+02],
                                       [-2.02685189e+02, -1.86624158e+02],
                                       [-1.98685189e+02, -1.96754158e+02],
                                       [-1.92555189e+02, -2.12264158e+02],
                                       [-1.88925189e+02, -2.16264158e+02],
                                       [-1.74305189e+02, -2.23144158e+02],
                                       [-1.74305189e+02, -2.23144158e+02],
                                       [-1.62185189e+02, -2.28904158e+02],
                                       [-1.62185189e+02, -2.28904158e+02],
                                       [-1.62185189e+02, -2.28904158e+02],
                                       [-1.51805189e+02, -2.28024158e+02],
                                       [-1.51805189e+02, -2.28024158e+02],
                                       [-1.29175189e+02, -2.26274158e+02],
                                       [-1.14555189e+02, -2.15644158e+02],
                                       [-1.06805189e+02, -1.95254158e+02],
                                       [-1.01805189e+02, -1.82374158e+02],
                                       [-1.01555189e+02, -1.73364158e+02],
                                       [-1.05675189e+02, -1.64234158e+02],
                                       [-1.07305189e+02, -1.60734158e+02],
                                       [-1.09055189e+02, -1.56354158e+02],
                                       [-1.09555189e+02, -1.54354158e+02],
                                       [-1.10055189e+02, -1.52354158e+02],
                                       [-1.13555189e+02, -1.47474158e+02],
                                       [-1.17425189e+02, -1.43474158e+02],
                                       [-1.23185189e+02, -1.37464158e+02],
                                       [-1.26055189e+02, -1.35594158e+02],
                                       [-1.34555189e+02, -1.32344158e+02],
                                       [-1.40055189e+02, -1.30214158e+02],
                                       [-1.48055189e+02, -1.28214158e+02],
                                       [-1.52185189e+02, -1.27834158e+02],
                                       [-1.58685189e+02, -1.27334158e+02],
                                       [-1.61185189e+02, -1.27834158e+02],
                                       [-1.70925189e+02, -1.31714158e+02],
                                       [-1.70925189e+02, -1.31714158e+02],
                                       [1.41944811e+02, -2.77741581e+01],
                                       [1.33814811e+02, -3.00241581e+01],
                                       [1.20194811e+02, -4.21541581e+01],
                                       [1.15694811e+02, -5.11641581e+01],
                                       [1.07194811e+02, -6.82941581e+01],
                                       [1.06694811e+02, -7.79341581e+01],
                                       [1.13574811e+02, -9.41941581e+01],
                                       [1.17814811e+02, -1.04194158e+02],
                                       [1.18574811e+02, -1.05074158e+02],
                                       [1.29694811e+02, -1.14704158e+02],
                                       [1.36194811e+02, -1.20334158e+02],
                                       [1.39314811e+02, -1.22084158e+02],
                                       [1.44814811e+02, -1.23334158e+02],
                                       [1.54944811e+02, -1.25584158e+02],
                                       [1.59444811e+02, -1.30834158e+02],
                                       [1.63574811e+02, -1.44844158e+02],
                                       [1.64814811e+02, -1.49104158e+02],
                                       [1.67694811e+02, -1.52854158e+02],
                                       [1.74944811e+02, -1.60104158e+02],
                                       [1.86574811e+02, -1.71614158e+02],
                                       [1.92694811e+02, -1.74114158e+02],
                                       [2.09694811e+02, -1.74244158e+02],
                                       [2.22694811e+02, -1.74244158e+02],
                                       [2.32444811e+02, -1.71494158e+02],
                                       [2.39944811e+02, -1.65484158e+02],
                                       [2.47574811e+02, -1.59484158e+02],
                                       [2.47944811e+02, -1.58984158e+02],
                                       [2.54074811e+02, -1.49354158e+02],
                                       [2.60194811e+02, -1.39844158e+02],
                                       [2.61814811e+02, -1.32464158e+02],
                                       [2.60574811e+02, -1.19584158e+02],
                                       [2.60074811e+02, -1.13574158e+02],
                                       [2.58944811e+02, -1.10574158e+02],
                                       [2.55444811e+02, -1.05574158e+02],
                                       [2.52944811e+02, -1.01944158e+02],
                                       [2.50944811e+02, -9.84441581e+01],
                                       [2.50944811e+02, -9.76941581e+01],
                                       [2.50944811e+02, -9.33141581e+01],
                                       [2.28944811e+02, -7.60541581e+01],
                                       [2.23444811e+02, -7.60541581e+01],
                                       [2.19194811e+02, -7.60541581e+01],
                                       [2.13824811e+02, -7.34241581e+01],
                                       [2.12444811e+02, -7.06741581e+01],
                                       [1.98314811e+02, -4.10341581e+01],
                                       [1.98314811e+02, -4.10341581e+01],
                                       [1.90574811e+02, -3.56541581e+01],
                                       [1.86444811e+02, -3.27741581e+01],
                                       [1.81944811e+02, -2.95241581e+01],
                                       [1.80574811e+02, -2.85241581e+01],
                                       [1.77194811e+02, -2.60241581e+01],
                                       [1.49944811e+02, -2.53941581e+01],
                                       [1.41944811e+02, -2.77741581e+01],
                                       [1.41944811e+02, -2.77741581e+01],
                                       [-6.84251890e+01, -2.82741581e+01],
                                       [-8.10551890e+01, -3.27741581e+01],
                                       [-9.45551890e+01, -4.67841581e+01],
                                       [-9.79251890e+01, -5.90441581e+01],
                                       [-9.98051890e+01, -6.57941581e+01],
                                       [-9.98051890e+01, -8.39341581e+01],
                                       [-9.78051890e+01, -9.10641581e+01],
                                       [-9.48051890e+01, -1.02324158e+02],
                                       [-8.05551890e+01, -1.17454158e+02],
                                       [-6.85551890e+01, -1.22084158e+02],
                                       [-6.20551890e+01, -1.24464158e+02],
                                       [-3.93051890e+01, -1.24714158e+02],
                                       [-3.23051890e+01, -1.22454158e+02],
                                       [-2.56751890e+01, -1.20204158e+02],
                                       [-1.38051890e+01, -1.10954158e+02],
                                       [-1.01851890e+01, -1.05194158e+02],
                                       [-8.55518900e+00, -1.02694158e+02],
                                       [-6.18518900e+00, -9.90641581e+01],
                                       [-4.92518900e+00, -9.70641581e+01],
                                       [-1.67518900e+00, -9.21841581e+01],
                                       [7.48109966e-02, -7.66741581e+01],
                                       [-1.42518900e+00, -6.60441581e+01],
                                       [-3.68518900e+00, -4.91641581e+01],
                                       [-1.83051890e+01, -3.16541581e+01],
                                       [-3.34251890e+01, -2.76441581e+01],
                                       [-4.24251890e+01, -2.52741581e+01],
                                       [-6.08051890e+01, -2.56441581e+01],
                                       [-6.84251890e+01, -2.82741581e+01],
                                       [-6.84251890e+01, -2.82741581e+01],
                                       [9.46948110e+01, 1.83735842e+02],
                                       [8.10748110e+01, 1.78985842e+02],
                                       [7.31948110e+01, 1.73105842e+02],
                                       [6.35748110e+01, 1.60595842e+02],
                                       [5.80748110e+01, 1.53345842e+02],
                                       [5.70748110e+01, 1.49345842e+02],
                                       [5.70748110e+01, 1.34705842e+02],
                                       [5.71948110e+01, 1.16695842e+02],
                                       [5.88148110e+01, 1.12945842e+02],
                                       [7.28148110e+01, 9.85558419e+01],
                                       [7.76948110e+01, 9.35558419e+01],
                                       [8.13148110e+01, 9.13058419e+01],
                                       [8.96948110e+01, 8.79258419e+01],
                                       [9.89448110e+01, 8.42958419e+01],
                                       [1.01314811e+02, 8.39258419e+01],
                                       [1.09074811e+02, 8.42958419e+01],
                                       [1.22574811e+02, 8.50458419e+01],
                                       [1.34944811e+02, 9.15558419e+01],
                                       [1.44574811e+02, 1.02815842e+02],
                                       [1.53074811e+02, 1.12565842e+02],
                                       [1.54444811e+02, 1.16445842e+02],
                                       [1.55444811e+02, 1.31575842e+02],
                                       [1.56324811e+02, 1.44965842e+02],
                                       [1.53694811e+02, 1.54215842e+02],
                                       [1.45574811e+02, 1.65605842e+02],
                                       [1.40574811e+02, 1.72605842e+02],
                                       [1.38444811e+02, 1.74355842e+02],
                                       [1.29444811e+02, 1.78985842e+02],
                                       [1.20194811e+02, 1.83865842e+02],
                                       [1.17944811e+02, 1.84485842e+02],
                                       [1.09074811e+02, 1.84865842e+02],
                                       [1.03194811e+02, 1.85115842e+02],
                                       [9.73148110e+01, 1.84735842e+02],
                                       [9.46948110e+01, 1.83735842e+02],
                                       [9.46948110e+01, 1.83735842e+02],
                                       [-1.69685189e+02, 2.32895842e+02],
                                       [-1.72425189e+02, 2.31515842e+02],
                                       [-1.76805189e+02, 2.29015842e+02],
                                       [-1.79425189e+02, 2.27395842e+02],
                                       [-1.84685189e+02, 2.24135842e+02],
                                       [-1.95305189e+02, 2.14255842e+02],
                                       [-1.95305189e+02, 2.12505842e+02],
                                       [-1.95305189e+02, 2.11885842e+02],
                                       [-1.97055189e+02, 2.07125842e+02],
                                       [-1.99185189e+02, 2.01875842e+02],
                                       [-2.02685189e+02, 1.93115842e+02],
                                       [-2.02925189e+02, 1.91245842e+02],
                                       [-2.02425189e+02, 1.82235842e+02],
                                       [-2.02055189e+02, 1.74235842e+02],
                                       [-2.00925189e+02, 1.70105842e+02],
                                       [-1.97305189e+02, 1.61595842e+02],
                                       [-1.92925189e+02, 1.51715842e+02],
                                       [-1.92055189e+02, 1.50595842e+02],
                                       [-1.84805189e+02, 1.45715842e+02],
                                       [-1.69175189e+02, 1.35085842e+02],
                                       [-1.51805189e+02, 1.32455842e+02],
                                       [-1.35305189e+02, 1.38085842e+02],
                                       [-1.25555189e+02, 1.41585842e+02],
                                       [-1.14175189e+02, 1.51465842e+02],
                                       [-1.09555189e+02, 1.60595842e+02],
                                       [-1.07685189e+02, 1.64355842e+02],
                                       [-1.05675189e+02, 1.66605842e+02],
                                       [-1.04305189e+02, 1.66605842e+02],
                                       [-1.03055189e+02, 1.66605842e+02],
                                       [-9.71851890e+01, 1.61595842e+02],
                                       [-9.11851890e+01, 1.55345842e+02],
                                       [-8.51851890e+01, 1.49215842e+02],
                                       [-7.75551890e+01, 1.42335842e+02],
                                       [-7.41751890e+01, 1.40335842e+02],
                                       [-7.08051890e+01, 1.38205842e+02],
                                       [-6.78051890e+01, 1.35705842e+02],
                                       [-6.74251890e+01, 1.34835842e+02],
                                       [-6.68051890e+01, 1.33205842e+02],
                                       [-7.61851890e+01, 1.24075842e+02],
                                       [-8.35551890e+01, 1.18945842e+02],
                                       [-8.63051890e+01, 1.17195842e+02],
                                       [-9.40551890e+01, 1.07565842e+02],
                                       [-9.63051890e+01, 1.03315842e+02],
                                       [-9.89251890e+01, 9.83058419e+01],
                                       [-1.00305189e+02, 8.32958419e+01],
                                       [-9.90551890e+01, 7.35458419e+01],
                                       [-9.76851890e+01, 6.30358419e+01],
                                       [-9.73051890e+01, 6.20358419e+01],
                                       [-9.04251890e+01, 5.21558419e+01],
                                       [-8.59251890e+01, 4.56458419e+01],
                                       [-7.81751890e+01, 3.93958419e+01],
                                       [-6.96851890e+01, 3.50158419e+01],
                                       [-6.44251890e+01, 3.22658419e+01],
                                       [-4.86851890e+01, 3.11358419e+01],
                                       [-3.91751890e+01, 3.26458419e+01],
                                       [-2.15551890e+01, 3.55158419e+01],
                                       [-4.30518900e+00, 5.27758419e+01],
                                       [-1.55518900e+00, 7.06658419e+01],
                                       [1.82481100e+00, 9.26758419e+01],
                                       [-5.05518900e+00, 1.10695842e+02],
                                       [-2.21851890e+01, 1.23695842e+02],
                                       [-2.69251890e+01, 1.27205842e+02],
                                       [-3.13051890e+01, 1.31205842e+02],
                                       [-3.19251890e+01, 1.32325842e+02],
                                       [-3.33051890e+01, 1.34705842e+02],
                                       [-3.30551890e+01, 1.34955842e+02],
                                       [-2.09251890e+01, 1.44085842e+02],
                                       [-1.30551890e+01, 1.50095842e+02],
                                       [-4.30518900e+00, 1.60845842e+02],
                                       [-2.92518900e+00, 1.66475842e+02],
                                       [3.44481100e+00, 1.90865842e+02],
                                       [-4.42518900e+00, 2.15505842e+02],
                                       [-2.19251890e+01, 2.26265842e+02],
                                       [-3.54251890e+01, 2.34645842e+02],
                                       [-3.54251890e+01, 2.34645842e+02],
                                       [-4.84251890e+01, 2.35145842e+02],
                                       [-5.96851890e+01, 2.35525842e+02],
                                       [-6.14251890e+01, 2.35275842e+02],
                                       [-6.61851890e+01, 2.32645842e+02],
                                       [-6.90551890e+01, 2.31015842e+02],
                                       [-7.41751890e+01, 2.28265842e+02],
                                       [-7.74251890e+01, 2.26395842e+02],
                                       [-8.13051890e+01, 2.24385842e+02],
                                       [-8.64251890e+01, 2.19635842e+02],
                                       [-9.09251890e+01, 2.14005842e+02],
                                       [-1.04425189e+02, 1.97625842e+02],
                                       [-1.04305189e+02, 1.97745842e+02],
                                       [-1.09175189e+02, 2.07625842e+02],
                                       [-1.13305189e+02, 2.16005842e+02],
                                       [-1.20425189e+02, 2.23515842e+02],
                                       [-1.30425189e+02, 2.29765842e+02],
                                       [-1.38305189e+02, 2.34645842e+02],
                                       [-1.38685189e+02, 2.34775842e+02],
                                       [-1.51555189e+02, 2.35145842e+02],
                                       [-1.62685189e+02, 2.35395842e+02],
                                       [-1.65425189e+02, 2.35025842e+02],
                                       [-1.69685189e+02, 2.32895842e+02],
                                       [-1.69685189e+02, 2.32895842e+02]]),
                             np.array([1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       79, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 79, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       4, 79]))

    # the constructor that loads the json file
    def __init__(self, anim_param_json_path):
        with open(anim_param_json_path) as file:
            data = json.load(file)

        # load in dicts and remove entries with none, also save in usable data type
        # using timedelta for durations and datetime for datetime
        self.var_paths = {key: (value) for key, value in data["paths"].items() if value}
        self.const_path = data["const_path"]
        self.var_time_stepping = data["data_time_stepping"]
        self.var_time_stepping = {key: parse_duration(value) for key, value in self.var_time_stepping.items() if value}
        self.vis_duration = parse_duration(data["vis_duration"])
        self.vis_start_date = datetime.fromisoformat(data["vis_start_date"])
        self.outputPath = data["outputPath"]
        self.enable_plotting = data["enable_plotting"]
        self.plotting_interval = data["plotting_interval"]
