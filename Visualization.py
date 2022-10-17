import concurrent.futures
import copy
import datetime
import gc
import time

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
from django.utils.dateparse import parse_duration

from VisParam import VisParam


class Visualization:
    """
        creates the visualization
        initialize with vis = Visualization(path_to_jpeg_file)
        and then call vis.visualize() to run
        this class assumes that the COSMO input files are named in the follwing way:
            "lffd" + YYYYmmddHHMMSS + ".nc"

            :parameter

            :returns
                jpg files which show the visualization
        """

    def __init__(self, anim_param_json_path):
        self.vp = VisParam(anim_param_json_path)
        # load const params
        (self.fr_land, self.for_e, self.for_d, self.alb_dif, self.grass_mask, self.grass_greenness, self.forest_mask,
         self.forest_greenness, self.elev, self.soiltyp, self.rotated_pole,
         self.native_resolution, self.lat_long_extend) = self.load_const_data(
            self.vp.const_path)
        self.outputFolder = self.vp.outputPath

        if self.vp.use_native_resolution:
            self.vp.figsize = self.native_resolution

    def visualize(self):
        vts = self.vp.var_time_stepping
        dt = self.find_min_time_step()
        data_read_intervals = self.data_read_intervals(dt, vts)
        print(f"Data read Intervals: {data_read_intervals}")
        num_time_steps = int(self.vp.vis_duration / dt)
        current_time = self.vp.vis_start_date
        print(f"Start time: {current_time}")
        to_plot = {key: None for key, value in data_read_intervals.items()}
        plotting_frequency = parse_duration(self.vp.plotting_interval) / dt

        skip = False

        # TODO implement multicore:
        #   idea is implemented but to_plot needs to be handled properly
        #   it gets overwritten and we get OOM-exceptions at runtime
        #   I disabled it for now
        #   Maybe a proper queue needs to be implemented that the main thread doesn't just run ahead
        workers = 12

        # this defines a possible multicore implementation, but it's disabled due to race conditions
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            tic = time.perf_counter()
            # futures = {}
            # queue_size = workers
            # print('Number of processes working: ' + str(executor.__getattribute__('_max_workers')))
            # main time loop
            for t in range(num_time_steps):
                for i in data_read_intervals:

                    # read data if there's new data available
                    if t % data_read_intervals[i] == 0:
                        time_str = self.get_timestring(current_time)
                        path = self.vp.var_paths[i] + "/lffd" + time_str + ".nc"

                        # if it's not plotting time
                        # concatenate multiple data reads together ( used for hail, lightning and precipitation)
                        # skip to next timestep
                        if plotting_frequency > data_read_intervals[i] and t % plotting_frequency != 0:
                            to_plot[i] = xr.concat([to_plot[i], self.load_variable(i, path)], dim='time')
                            skip = True
                            continue

                        elif t == 0:
                            to_plot[i] = self.load_variable(i, path)

                if skip:
                    current_time += dt
                    skip = False
                    # print('skipped')
                    continue

                time_str = self.get_timestring(current_time)
                data_time = time.perf_counter() - tic

                # # part of the test multicore that is commented out for the moment
                # tmp = to_plot
                # limit the queue to limit memory consumption
                # while len(futures) > queue_size:
                #     # wait for queue to be smaller
                #     a = 1
                # idx = executor.submit(self.plot_all, current_time, tmp)
                # del tmp
                # futures[idx] = t
                print(f"data read time {time_str}: {data_time:0.4f}s")
                self.plot_all(current_time, to_plot)

                # # part of the test multicore that is commented out for the moment
                # for future in futures.copy():
                #     if future.done():
                #         tmp = future.result()
                #         del futures[future]

                gc.collect()

                for i in data_read_intervals:

                    if t % data_read_intervals[i] == 0:
                        time_str = self.get_timestring(current_time)
                        path = self.vp.var_paths[i] + "/lffd" + time_str + ".nc"

                        to_plot[i] = self.load_variable(i, path)

                current_time += dt
                tic = time.perf_counter()
                gc.collect()

    def plot_all(self, current_time, to_plot):
        time_str = self.get_timestring(current_time)
        tic = time.perf_counter()
        fig, ax = plt.subplots(figsize=self.vp.figsize, subplot_kw={"projection": self.rotated_pole, "frameon": False})
        #ax.set_extent(self.lat_long_extend, crs=ccrs.PlateCarree())
        # ax.set_global()

        if self.vp.draw_grid_lines:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=2, color='gray', alpha=0.5, linestyle='--', x_inline=False,
                              y_inline=False, xformatter=LongitudeFormatter(),
                              yformatter=LatitudeFormatter())
            gl.xlabel_style = {'size': self.vp.grid_labelsize, 'color': 'black'}
            gl.ylabel_style = {'size': self.vp.grid_labelsize, 'color': 'black'}

        # plot the variables in the correct order

        if self.vp.enable_plotting["W_SO"]:
            self.plot_land_ocean(to_plot["W_SO"], ax)
            self.plot_var("elev", self.elev, self.vp.cmap_elev, ax, only_plot_on_land=True)
            self.plot_var("ice", self.soiltyp, self.vp.cmap_snow, ax)

        if self.vp.enable_plotting["W_SNOW"]:
            self.plot_var("W_SNOW", to_plot["W_SNOW"].mean('time'), self.vp.cmap_snow, ax)

        if self.vp.enable_plotting["TQV"]:
            self.plot_var("TQV", to_plot["TQV"].mean('time'), self.vp.cmap_TQV, ax, only_plot_on_ocean=True)

        if self.vp.enable_plotting["TQC"]:
            self.plot_var("TQC", to_plot["TQC"].mean('time'), self.vp.cmap_TQC, ax)

        if self.vp.enable_plotting["TQI"]:
            self.plot_var("TQI", to_plot["TQI"].mean('time'), self.vp.cmap_TQI, ax)

        if self.vp.enable_plotting["TOT_PREC"]:
            self.plot_var("TOT_PREC", to_plot["TOT_PREC"].sum('time'), self.vp.cmap_TOT_PREC, ax)

        if self.vp.enable_plotting["DHAIL_MX"]:
            self.plot_marker("DHAIL_MX", to_plot["DHAIL_MX"].max('time'), ax)

        if self.vp.enable_plotting["LPI"]:
            self.plot_marker("LPI", to_plot["LPI"].max('time'), ax)

        # clear the title and plot time at the top write corner
        ax.set_title('')
        ax.set_title('{:%d.%m.%Y %H:00}'.format(current_time), loc='right',
                     fontsize=self.vp.time_font_size,
                     bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 1})

        picpath = self.outputFolder + "/" + time_str + ".jpg"
        plt.savefig(picpath, dpi=self.vp.dpi)
        toc = time.perf_counter()
        print(f"render time for frame {time_str}: {toc - tic:0.4f}s", flush=True)
        # close plotted figure to avoid memory leak
        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')
        plt.close(fig)
        gc.collect()
        return 0

    @staticmethod
    def load_const_data(filename):
        # loading all relevant constant fields
        print("loading data from:", filename)
        # loading all relevant constant fields
        with xr.open_dataset(filename) as ds:
            # ds = select_domain(ds, vp.plot_domain)
            # ds = ds.rename({'rlon': 'lon', 'rlat': 'lat'})
            elev = ds.HSURF.load()
            soiltyp = ds.SOILTYP.load()
            ice = soiltyp.where(soiltyp == 1)
            fr_land = ds.FR_LAND.load()

            # fr_land[0, :, :].squeeze().plot()
            for_e = ds.FOR_E.load()
            for_d = ds.FOR_D.load()
            alb_dif = ds.ALB_DIF.load()
            rot_pole = ds.rotated_pole.load()
            lat = ds.lat.load()
            lon = ds.lon.load()
        # get rotated pole
        rotated_pole = ccrs.RotatedPole(pole_longitude=rot_pole.grid_north_pole_longitude,
                                        pole_latitude=rot_pole.grid_north_pole_latitude)

        # trying to make white box around plot smaller / unssuccesful so far
        min_lat = min([lat.data[0, 0], lat.data[0, -1]], key=abs)
        max_lat = min([lat.data[-1, 0], lat.data[-1, -1]], key=abs)
        min_lon = min([lon.data[0, 0], lon.data[-1, 0]], key=abs)
        max_lon = min([lon.data[0, -1], lon.data[-1, -1]], key=abs)

        lat_long_extend = [min_lon, max_lon, min_lat, max_lat]
        #print(lat_long_extend)

        px = 1. / 100.  # pixel in inches
        native_resolution = (for_e.sizes.get('rlon') * px, for_e.sizes.get('rlat') * px)

        fr_land = fr_land.mean(dim='time')
        for_e = for_e.mean(dim='time')
        for_d = for_d.mean(dim='time')
        alb_dif = alb_dif.mean(dim='time')

        # land mask
        grass_mask = fr_land.copy()
        grass_mask *= (1 - for_e.copy())
        grass_mask *= (1 - for_d.copy())
        grass_greenness = grass_mask * (1 - alb_dif.copy())

        # forest mask
        forest_mask = for_e + for_d
        forest_greenness = forest_mask * (1 - alb_dif)

        return fr_land, for_e, for_d, alb_dif, grass_mask, grass_greenness, forest_mask, forest_greenness, elev, ice, rotated_pole, native_resolution, lat_long_extend

    @staticmethod
    def load_variable(var_name, path):
        """

        :param var_name: name/type of variable
        :param path: the path to the file
        :return: variable var_name as an xarray data array
        """
        with xr.open_dataset(path) as ds:
            return ds[var_name].load()

    def plot_land_ocean(self, wsoil, ax):

        # exact number of the deepest saturation layer for grass
        dz_grass = max([x for x in wsoil.indexes['soil1'] if x <= self.vp.deepest_soil_level_grass])
        # exact number of the deepest saturation layer for forest
        dz_forest = max([x for x in wsoil.indexes['soil1'] if x <= self.vp.deepest_soil_level_forest])

        # select W_SO of the specified soil lvls
        wsoil_grass = copy.copy(wsoil.sel(soil1=slice(0.0, dz_grass)))
        wsoil_forest = copy.copy(wsoil.sel(soil1=slice(0.0, dz_forest)))

        # sum over all chose soil lvls
        wsoil_grass = wsoil_grass.sum(dim='soil1')
        wsoil_forest = wsoil_forest.sum(dim='soil1')

        # take the mean over time
        wsoil_grass = wsoil_grass.mean(dim='time')
        wsoil_forest = wsoil_forest.mean(dim='time')

        # calculate the soil saturation
        sat_soil_grass = wsoil_grass / (
                self.vp.saturation_percentage * dz_grass)
        sat_soil_forest = wsoil_forest / (
                self.vp.saturation_percentage * dz_forest)

        max_growth = 1.0
        plant_growth_grass = self.grass_mask * sat_soil_grass / self.vp.land_plant_max_growth_sat_soil
        plant_growth_grass.values[plant_growth_grass.values > max_growth] = max_growth
        plant_growth_forest = self.forest_mask * sat_soil_forest / self.vp.forest_plant_max_growth_sat_soil
        plant_growth_forest.values[plant_growth_forest.values > max_growth] = max_growth

        grass_greenness = self.grass_greenness * plant_growth_grass
        forest_greenness = self.forest_greenness * plant_growth_forest

        forest_greenness.values[
            (self.forest_mask.values > 0.3) &
            (forest_greenness.values < self.vp.min_forest_greenness)] = \
            self.vp.min_forest_greenness

        land_greenness = grass_greenness + forest_greenness

        # plot land
        land_greenness.squeeze(). \
            plot.pcolormesh(
            ax=ax,
            cmap=self.vp.cmap_land,
            vmin=0.0, vmax=0.9,
            add_colorbar=False, add_labels=False)

        # plot ocean
        ocean_mask = (1 - self.fr_land)  # * tmpfct
        ocean_mask.squeeze(). \
            plot.pcolormesh(
            ax=ax,
            cmap=self.vp.cmap_ocean,
            add_colorbar=False, add_labels=False)

    def plot_var(self, var_name, var, cmap, ax, only_plot_on_ocean=False, only_plot_on_land=False):
        """
        plots variable at a given point in time
        :param ax:
        :param only_plot_on_land: boolean
        :param var: the DataArray of the variable to plot itself
        :param cmap: colormap for the variable to plot
        :param only_plot_on_ocean: boolean
        :param var_name: name of the variable e.g. TQC, TQI, W_SO
        :return:
        """

        if only_plot_on_ocean:
            ocean_mask = 1 - self.fr_land
            ocean_mask['lon'] = var.lon
            ocean_mask['lat'] = var.lat
            var = var.where(ocean_mask, np.nan)

        if only_plot_on_land:
            land_mask = self.fr_land
            land_mask['lon'] = var.lon
            land_mask['lat'] = var.lat
            var = var.where(land_mask, np.nan)

        if var_name == "TOT_PREC" and self.vp.do_precipitation_colourbar:
            var.squeeze(). \
                plot.pcolormesh(ax=ax, cmap=cmap, vmin=self.vp.variable_plotting_bounds[var_name][0],
                                vmax=self.vp.variable_plotting_bounds[var_name][1], add_colorbar=True,
                                add_labels=False,
                                rasterized=True,
                                linewidth=0, edgecolor='face',
                                antialiased=True)

        else:
            var.squeeze(). \
                plot.pcolormesh(ax=ax, cmap=cmap, vmin=self.vp.variable_plotting_bounds[var_name][0],
                                vmax=self.vp.variable_plotting_bounds[var_name][1], add_colorbar=False,
                                add_labels=False,
                                rasterized=True,
                                linewidth=0, edgecolor='face',
                                antialiased=True)

    def plot_marker(self, var_name, var, ax):
        """
        plots highly local variable at a given point in time e.g. hail / lighning
        :param ax:
        :param var: the variable itself
        :param var_name: name of the variable e.g. TQC, TQI, W_SO
        :return:
        """

        # omit values less than 0 and use a coarser grid for markers
        var = (var.where(var > 0)).to_dataset()
        var = var.coarsen(rlon=20, rlat=20, boundary='pad').max()

        vmin = self.vp.variable_plotting_bounds[var_name][0]  # assign here to avoid long call in function below
        vmax = self.vp.variable_plotting_bounds[var_name][1]

        if var_name == "LPI":
            xr.plot.scatter(var, ax=ax, x='rlon', y='rlat', hue=var_name, marker=self.vp.marker_lightning,
                            add_guide=False, s=200, cmap=self.vp.cmap_black, linewidths=2, plotnonfinite=False,
                            levels=np.linspace(vmin, vmax, 1),
                            transform=self.rotated_pole, label=None)

            xr.plot.scatter(var, ax=ax, x='rlon', y='rlat', hue=var_name, marker=self.vp.marker_lightning,
                            add_guide=False, s=200, cmap=self.vp.cmap_white, linewidths=0.7, plotnonfinite=False,
                            levels=np.linspace(vmin, vmax, 1),
                            transform=self.rotated_pole, label=None)

        elif var_name == "DHAIL_MX":
            xr.plot.scatter(var, ax=ax, x='rlon', y='rlat', hue=var_name, marker=self.vp.marker_hail,
                            add_guide=False, s=100, cmap=self.vp.cmap_black, linewidths=0.5, plotnonfinite=False,
                            levels=np.linspace(vmin, vmax, 1),
                            transform=self.rotated_pole, label=None)
        else:
            xr.plot.scatter(var, ax=ax, x='rlon', y='rlat', hue=var_name, marker='x',
                            add_guide=False, s=200, cmap=self.vp.cmap_black, linewidths=2, plotnonfinite=False,
                            levels=np.linspace(vmin, vmax, 1),
                            transform=self.rotated_pole, label=None)

    def find_min_time_step(self):
        # if not specified use 30 minutes as minimum time stepping
        if len(self.vp.var_time_stepping.values()) == 0:
            return datetime.timedelta(minutes=30)
        return min(self.vp.var_time_stepping.values())

    @staticmethod
    def data_read_intervals(timestep, vts):
        """
        calculates in which interval certain variables get plotted
        does not work generally: smallest timestep has to be a denominator of the others
        :param timestep: time stepping for the animation
        :param vts: time stepping dictionary
        :return:
        """
        return {key: int(value / timestep) for key, value in vts.items() if value}

    @staticmethod
    def get_timestring(current_time):
        return "{:%Y%m%d%H%M%S}".format(current_time)
