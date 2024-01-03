"""
to assess wind resource in Reunion
"""

__version__ = f'Version 2.0  \nTime-stamp: <2021-05-15>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import hydra
import matplotlib
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from importlib import reload
from omegaconf import DictConfig
# import geopandas as gpd
import GEO_PLOT
import DATA


def jk():
    print(f'reloading GEO_PLOT....')
    reload(GEO_PLOT)


def voronoi(area: str = 'reunion', coords=[[]]):
    import geopandas as gpd

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    area = world[world.name == area]

    area = area.to_crs(epsg=3395)  # convert to World Mercator CRS
    area_shape = area.iloc[0].geometry  # get the Polygon
    # Now we can calculate the Voronoi regions, cut them with the
    #     geographic area shape and assign the points to them:

    from geovoronoi import voronoi_regions_from_coords

    region_polys, region_pts = voronoi_regions_from_coords(coords, area_shape)

    from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

    fig, ax = subplot_for_map()
    plot_voronoi_polys_with_points_in_area(ax, area_shape, region_polys, coords, region_pts)
    plt.show()


# ----------------------------- functions -----------------------------


@hydra.main(version_base='1.3', config_path="configs", config_name="wind_reu_config")
def wind_resource(cfg: DictConfig) -> None:
    """
    to find the physical link between the SSR classification and the large-scale variability
    over la reunion island
    """

    # calculate necessary data and save it for analysis:
    # ==================================================================== data:

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job))):

        if cfg.job.voronoi:
            # read station data
            station = pd.read_csv(f'{cfg.dir.local_data:s}/MF/{cfg.data.mf_station:s}')

            coords = np.array(station[['LON', 'LAT']])

            alt = np.array(station['ALT'])

            # plot voronoi:

            from matplotlib.collections import PolyCollection
            from scipy.spatial import Voronoi, voronoi_plot_2d
            vor = Voronoi(coords)

            # ---------------------------------------
            # Plot Voronoi diagram with filled color
            def voronoi_finite(vor):
                """
                Reconstruct infinite Voronoi regions in a finite space.

                Parameters:
                vor : scipy.spatial.Voronoi
                    Voronoi diagram object.

                Returns:
                regions : list of list of tuple
                    List of finite Voronoi regions.
                vertices : array
                    Voronoi diagram vertices.
                """
                new_regions = []
                for region in vor.regions:
                    if -1 not in region and len(region) > 0:
                        new_regions.append([(vor.vertices[i, 0], vor.vertices[i, 1]) for i in region])
                return new_regions, vor.vertices

            fig, ax = plt.subplots(figsize=(10, 8))

            # plot filled color:
            regions, vertices = voronoi_finite(vor)
            polygons = PolyCollection(regions, edgecolor='black', cmap='viridis')
            polygons.set_array(alt)
            ax.add_collection(polygons)

            # plot lines and points:
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange',
                            line_width=2, line_alpha=0.6, point_size=8,
                            figsize=(8, 8), dpi=220)

            # Customize the colorbar
            cbar = plt.colorbar(polygons, ax=ax)
            cbar.set_label('Altitude (meter)')

            # add axis labels:
            plt.xlabel('Longitude ($^\circ$E)', fontsize=12)
            plt.ylabel('Latitude ($^\circ$N)', fontsize=12)

            # Customize the plot as needed
            ax.set_xlim(vor.min_bound[0]-0.05, vor.max_bound[0]+0.05)
            ax.set_ylim(vor.min_bound[1]-0.05, vor.max_bound[1]+0.05)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('Voronoi Diagram with MF stations with ALT in color')

            # add coastline:
            coastline = GEO_PLOT.load_reunion_coastline()
            plt.scatter(coastline.longitude, coastline.latitude, marker='o', s=1, c='gray', edgecolor='gray', alpha=0.6)
            plt.savefig(cfg.figure.reunion_voronoi_mf, dpi=300, bbox_inches='tight')
            plt.show()
            # =====

        print('working')


if __name__ == "__main__":
    sys.exit(wind_resource())
