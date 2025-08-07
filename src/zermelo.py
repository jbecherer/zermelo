import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyproj
import time

class Spoint:
    """
    Class representing a point in space with coordinates and heading.
    Attributes:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        time (datetime): time of the point.
        r (float): radial distance from the origin.
        phi (float): angle in radians from current position towards the target(origin).
        Halpha (list): history of headings.
        Hpos (list): history of positions.

    Methods:
        __init__(X, Halpha=None, Hpos=None): Initializes the Spoint with coordinates and optional history.
        __repr__(): Returns a string representation of the Spoint.
        __add__(other): Adds two Spoint objects together by adding their coordinates.
        __sub__(other): Subtracts two Spoint objects by subtracting their coordinates.
    """
    def __init__(self, X, t=None, Halpha=None, Hpos=None):
        self.x = X[0]
        self.y = X[1]
        #polar coordinates
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.phi = np.arctan2(self.y, self.x)

        if t is None:
            self.time = np.datetime64('now')
        else:
            self.time = t

        # history of headings 
        if Halpha is None:
            self.Halpha = []
        else:
            self.Halpha = Halpha

        # history of positions
        if Hpos is None:
            self.Hpos = []
        else:
            self.Hpos = Hpos
        self.Hpos.append(X)

    def __repr__(self):
        return f"Spoint(x={self.x}, y={self.y}, r={self.r}, phi={self.phi}, Halpha={self.Halpha}, Hpos={self.Hpos})"

    def __add__(self, other):
        """
        Adds two Spoint objects together by adding their coordinates.
        """
        return Spoint([self.x + other.x, self.y + other.y], self.time, self.Halpha, self.Hpos)
    def __sub__(self, other):
        """        Subtracts two Spoint objects by subtracting their coordinates.
        """
        return Spoint([self.x - other.x, self.y - other.y], self.time, self.Halpha, self.Hpos)




def zermelo_solution(A, B, u, v, w, start_time, dt, dalpha, dl):
    """
    Calculates the Zermelo solution for a given set of parameters.

    Inputs:
        A (array (lon,lat)):  coordinates for the starting point.
        B (array (lon,lat)):  coordinates for the destination point.
        u (xarray): x-component of the current velocity field.
        v (xarray): y-component of the current velocity field.
        w (float): speed of glider (default=0.3 m/s)
        start_time (datetime): time of the starting point (default is the first time in u)
        dt (float): time of dives (default=6 hours)
        dalpha (float): angle resolution of the heading (heading resolution in radians, default=pi/18)
        dl (float): horizontal resolution of the solution line (default= 0.1 w * dt)

    Outputs:
        X [arrax(x,y)]: x-coordinates of the solution line.
        alpha (array): headings of the solution line.
        time (datetime): time of the solution line.
    """

    # set defaults
    if dalpha is None:
        dalpha = np.pi / 9  # 10 degrees
    if dt is None:
        dt = 3600 * 6 # 6 hour
    if w is None:
        w = .3 # 0.3 m/s
    if dl is None:
        dl = 0.3 * w * dt

    if start_time is None:
        start_time = u.time.values[0]

    # target condition
    r_target = w * dt

    A = np.array(latlon_to_gauss_krueger(A[0], A[1], B))  
    # list of possible points at each time step
    P = Spoint(A, t=start_time)  # starting point

    Ps = [P]  # list of points in the tree

    for i in range(1000):
        # propagation
        Ps = propaget_tree(Ps, u, v, w, dt, dalpha, B)

        # test if any of the points hit the target condition
        rs = np.array([P.r for P in Ps])
        print(f"distance to B  {np.min(rs):.3f} m")
        if np.any(rs < r_target):
            print("Found points that hit the target condition")
            ind = np.argmin(rs)
            return Ps[ind].Hpos, Ps[ind].Halpha, Ps[ind].time

        # pruning 
        Ps = prune_tree(Ps, dl)
        # print(f"Number of points in the tree: {len(Ps)}")



    return None, None, None 


def prune_tree(Ps, dl):
    """
    Prunes the tree of points by removing points that are too close to each other in the same angle bin.
    Inputs:
        Ps (list of Spoint): list of points in the tree.
        dl (float): horizontal resolution of the solution line.
    Outputs:
        Ps_new (list of Spoint): updated list of points in the tree after pruning.
    """

    rs = np.array([P.r for P in Ps])
    r_mean = np.nanmean(rs)

    phis = np.array([P.phi for P in Ps])
    dphi = dl / r_mean

    indexes2delete = []

    
    for phi_bin in np.arange(-np.pi,  np.pi, dphi):
        # find all points in the same angle bin 
        indexes = np.where(np.abs(phis - phi_bin) < dphi / 2)[0]
        if len(indexes) > 0:
            # find the point closest to the target
            r_diffs = rs[indexes]
            min_index = indexes[np.argmin(r_diffs)]
            # delete all other points in the same angle bin
            indexes2delete.extend(indexes[indexes != min_index])

    # remove indexes that are more than twice rs.min()
    indexes2delete_r = [i for i in range(len(rs)) if ((rs[i] > 1.5 * np.nanmin(rs)) or (np.isnan(rs[i])))]
    # print(f"Number of points to delete: {len(indexes2delete)} angle   {len(indexes2delete_r)} distance")
    # delete points
    Ps_new = [P for i, P in enumerate(Ps) if i not in indexes2delete and i not in indexes2delete_r]

    return Ps_new

def propagate_direction(P, u, v, w, dt, alpha, B):
    """
    Propagates a single point in a given direction based on the current velocity field.

    Inputs:
        P (Spoint): point to propagate.
        u (array): x-component of the current velocity field.
        v (array): y-component of the current velocity field.
        w (float): speed of glider.
        dt (float): time of dives.
        alpha (float): heading angle in radians.
        B (array): coordinates of the center point for Gauss-Krueger projection (longitude, latitude).

    Outputs:
        P_new (Spoint): new point generated from the propagation.
    """

    # calculate new position
    dx_glider = w * np.cos(alpha) * dt 
    dy_glider = w * np.sin(alpha) * dt 


    mid_x = P.x + dx_glider
    mid_y = P.y + dy_glider
    mid_lon, mid_lat = xy_to_latlon(mid_x, mid_y, B)
    u_mid = u.sel(latitude=mid_lat, longitude=mid_lon, time=P.time, method='nearest').values
    v_mid = v.sel(latitude=mid_lat, longitude=mid_lon, time=P.time, method='nearest').values
    if np.isnan(u_mid) or np.isnan(v_mid):
        # print(f"Warning: u or v is NaN at {mid_lon}, {mid_lat} at time {P.time}. Using zero velocity.")
        u_mid = 0.0
        v_mid = 0.0


    dx_current = u_mid * dt
    dy_current = v_mid * dt
    
    new_x = P.x + dx_glider + dx_current
    new_y = P.y + dy_glider + dy_current

    Haplpha = P.Halpha.copy()
    Haplpha.append(alpha)
    Hpos = P.Hpos.copy()
    Ptime = P.time.copy() + np.timedelta64(dt, 's')
    # create new point
    P_new = Spoint(X=[new_x, new_y], t=Ptime, Halpha=Haplpha, Hpos=P.Hpos.copy())
   
    # # update history
    # P_new.Halpha = P.Halpha + [alpha]
    # P_new.Hpos = P.Hpos + [[new_x, new_y]]

    return P_new

def propagate_point(P, u, v, w, dt, dalpha, B):
    """
    Propagates a single point based on the current velocity field and heading resolution.

    Inputs:
        P (Spoint): point to propagate.
        u (xarray): x-component of the current velocity field.
        v (xarray): y-component of the current velocity field.
        w (float): speed of glider.
        dt (float): time of dives.
        dalpha (float): angle resolution of the heading.
        B (array): coordinates of the center point for Gauss-Krueger projection (longitude, latitude).

    Outputs:
        Ps_new (list of Spoint): list of new points generated from the propagation.
    """

    Ps_new = []

    # iterate over possible headings
    phi = P.phi
    # for alpha in np.arange(0, 2 * np.pi, dalpha):
    if np.isnan(phi):
        return Ps_new  # if phi is NaN, return empty list
    for alpha in np.arange(-0.5*np.pi-phi, 0.5 * np.pi-phi, dalpha):
        new_point = propagate_direction(P, u, v, w, dt, alpha, B)
        Ps_new.append(new_point)

    return Ps_new



def propaget_tree(Ps, u, v, w, dt, dalpha, B):
    """
    Propagates the tree of points based on the current velocity field and heading resolution.

    Inputs:
        Ps (list of Spoint): list of points in the tree.
        u (xarray): x-component of the current velocity field.
        v (yarray): y-component of the current velocity field.
        w (float): speed of glider.
        dt (float): time of dives.
        dalpha (float): angle resolution of the heading.
        B (array): coordinates of the center point for Gauss-Krueger projection (longitude, latitude).

    Outputs:
        Ps (list of Spoint): updated list of points in the tree after propagation.
    """

    Ps_new = []


    for P in Ps:
        Ps_new.extend(propagate_point(P, u, v, w, dt, dalpha, B))

    return Ps_new 

def get_projection(B):
    """Returns a pyproj projection object for Gauss-Krueger projection centered on the specified longitude."""
    if B is None:
        lon_0 = -65.0  
        lat_0 = 18.5
    else:
        lon_0 = B[0]
        lat_0 = B[1]

    return  pyproj.Proj(proj='tmerc', lat_0=lat_0, lon_0=lon_0, k=1, x_0=0, y_0=0, ellps='WGS84')


def dataset_latlon_to_gauss_krueger(ds, B):
    """Converts latitude and longitude coordinates in an xarray dataset to Gauss-Krueger coordinates.
    Inputs:
        ds (xarray.Dataset): dataset with latitude and longitude coordinates.
        B (array): coordinates of the center point for Gauss-Krueger projection (longitude, latitude).
    Outputs:
        ds (xarray.Dataset): dataset with updated coordinates in Gauss-Krueger projection.
    """
    lon = ds.longitude.values
    lat = ds.latitude.values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = latlon_to_gauss_krueger(lon_grid, lat_grid, B)
    ds = ds.assign_coords(x=(('latitude', 'longitude'), x), y=(('latitude', 'longitude'), y))
    return ds

def latlon_to_gauss_krueger(lon, lat, B):
    """
    Converts latitude and longitude to Gauss-Krueger coordinates.

    Inputs:
        lat (array): array of latitudes.
        lon (array): array of longitudes.
        B (array): coordinates of the center point for Gauss-Krueger projection (longitude, latitude).

    Outputs:
        x (array): x-coordinates in Gauss-Krueger projection.
        y (array): y-coordinates in Gauss-Krueger projection.
    """
    proj = get_projection(B)
    x, y = proj(lon, lat)
    return x, y


def xy_to_latlon(x, y, B):
    """
    Converts Gauss-Krueger coordinates to latitude and longitude.

    Inputs:
        x (array): x-coordinates in Gauss-Krueger projection.
        y (array): y-coordinates in Gauss-Krueger projection.
        B (array): coordinates of the center point for Gauss-Krueger projection (longitude, latitude).

    Outputs:
        lon (array): longitudes.
        lat (array): latitudes.
    """
    proj = get_projection(B)
    lon, lat = proj(x, y, inverse=True)
    return lon, lat

def direct_flight(A, B, u,v, w, dt):
    """ Calculates a direct flight path from point A to point B, ignoring the current velocity field.
        Always pointing the heading towards the target point B.
    Inputs:
        A (array): coordinates for the starting point.
        B (array): coordinates for the destination point.
        u (xarray): x-component of the current velocity field.
        v (xarray): y-component of the current velocity field.
        w (float): speed of glider (default=0.3 m/s)
        dt (float): time of dives (default=6 hours)
    Outputs:
        X (list of arrays): list of coordinates of the solution line.
        alpha (list of floats): list of headings of the solution line.
    """

    A = np.array(latlon_to_gauss_krueger(A[0], A[1], B))  
    P = Spoint(A, u.time.values[0])  # starting point

    for i in range(1000):
        alpha = P.phi + np.pi # invert direction to the target
        # print(f"Step {i}: Current position: {P.x:.3f}, {P.y:.3f}, r={P.r:.3f}, phi={P.phi:.3f}, alpha={alpha:.3f}")
        P = propagate_direction(P, u, v, w, dt, alpha, B)
        # check if the point is close enough to the target
        if P.r < w * dt:
            return P.Hpos, P.Halpha

    return P.Hpos, P.Halpha


#==============================================================================
# extra functions
#==============================================================================


def load_velocity_field(B, data_dir):
    """Loads the geostrophic currents from a NetCDF file and returns the u and v components.
    Inputs:
        B (array): coordinates of the center point for Gauss-Krueger projection (longitude, latitude).
    data_dir :
    """
    # U_data = xr.open_dataset('../copernicus_data/data/obs_vel_2025-07-31.nc')
    data_dir = data_dir or "."
    path = os.path.join(data_dir, "model_latest.nc")
    U_data = xr.open_dataset(path)
    # pick only the last time step
    # U_data = U_data.isel(time=1)
    # depth average
    U_data = U_data.mean(dim='depth')

    U_data = dataset_latlon_to_gauss_krueger(U_data, B)

    min_lon = -70.0
    max_lon = -60.0
    min_lat = 17.0
    max_lat = 25.0

    U_data = U_data.sel(longitude=slice(min_lon, max_lon), latitude=slice(min_lat, max_lat))
    # check for variable names in xarray dataset
    if 'ugos' in U_data.variables.keys():
        u = U_data['ugos']
        v = U_data['vgos']
    else:
        u = U_data['uo']
        v = U_data['vo']

    return u, v
    
def plot_velocity_field(u, v, w):
    plt.close('all')
    # project u and v on polar coordinates directed towards the target point B

    fig, ax = plt.subplots(1,2, figsize=(10, 8), sharex=True, sharey=True)

    # trim velocityies to use only every 5th point in the lat and lon direction
    uq = u.isel(latitude=slice(None, None, 3), longitude=slice(None, None, 3))
    vq = v.isel(latitude=slice(None, None, 3), longitude=slice(None, None, 3))

    spd = u.copy()
    spd.values = np.sqrt(u**2 + v**2)
    # for i in range(len(u.latitude)):
    #     for j in range(len(u.longitude)):
    #         spd[0,i,j] = -1/w *(u[0,i,j]*u.x[i,j] + v[0,i,j]*u.y[i,j]) / np.sqrt(u.x[i,j]**2 + u.y[i,j]**2)
    #
    ia = [0, spd.shape[0]-1]
    for a in range(len(ax)):
        ax[a].set_aspect('equal')
        ax[a].set_xlabel('Longitude (degrees)')
        ax[a].set_ylabel('Latitude (degrees)')
        ax[a].pcolor(u.longitude, u.latitude, spd[ia[a],:,:], cmap='Blues', vmin=0, vmax=.7)
        # contour of spd = -1
        ax[a].contour(u.longitude, u.latitude, spd[ia[a],:,:], levels=[-1], colors='k', linewidths=0.5)
        # quiver plot of u and v
        ax[a].quiver(uq.longitude, uq.latitude, uq[ia[a],:,:], vq[ia[a],:,:], color=[.6,.6,.6], scale=5, width=0.002)
        ax[a].text(.1, .9, f'Time: {np.datetime_as_string(u.time[ia[a]].values, unit="D")}', transform=ax[a].transAxes)

    return fig, ax

def plot_solutions(ax, X, B, color='red', label='Zermelo solution'):
    """
    Plots the Zermelo solution on the given axes.
    
    Inputs:
        ax (matplotlib.axes.Axes): axes to plot on.
        X (list): coordinates of the solution line.
        B (array): coordinates of the target point.
    """
    X = np.array(X)
    lon, lat = xy_to_latlon(X[:, 0], X[:, 1], B)
    ax.plot(lon, lat, color=color, linewidth=4, label=label)


def get_waypoint(A, B, w=0.25, dt=3600*6, data_dir=""):
    """Calculates a waypoint between points A and B, 1/3 of the way from A to B.
    Inputs:
        A (array) [lon, lat]: coordinates for the starting point.
        B (array) [lon, lat]: coordinates for the destination point.
        w :
        dt:
        data_dir :
    Outputs:
        wp [lon, lat]: waypoint coordinates in longitude and latitude
        heading (float): heading to way point in radians
    """

    u, v = load_velocity_field(B, data_dir)
    X, alpha, times = zermelo_solution(A, B, u, v, w, np.datetime64('now'), dt, None, None)

    # cal waypoint 
    x, y = latlon_to_gauss_krueger(A[0], A[1], B)
    r = 20000 # 20 km waypoint
    wp_complex = r * np.exp(1j * alpha[0])  # complex number for waypoint
    lon, lat = xy_to_latlon(x + wp_complex.real, y + wp_complex.imag, B)
    wp = np.array([lon, lat])
    

    return wp , alpha[0]  # return waypoint and initial heading

def __main__(A, B, w=0.25, dt=3600*6, dalpha=np.pi/9, dl=None):

    u, v = load_velocity_field(B, '../copernicus_data/data/model_latest.nc')

    wp, _ = get_waypoint(A, B, w=w, dt=dt)
    X, alpha, times = zermelo_solution(A, B, u, v, w, np.datetime64('now'), dt, dalpha, dl)
    X_d, alpha_d = direct_flight(A, B, u, v, w, dt)


    fig, ax = plot_velocity_field(u, v, w)

    for a in range(len(ax)):
        plot_solutions(ax[a], X, B, color='green', label='Zermelo solution')
        plot_solutions(ax[a], X_d, B, color='red', label='Direct flight')
        ax[a].plot(B[0], B[1], 'mx', label='Target point B')
        ax[a].plot(A[0], A[1], 'ko', label='Start Point A')
        ax[a].plot(wp[0], wp[1], 'bo', label='Waypoint')

    ax[0].legend()
    ax[0].set_xlim([np.min([B[0], A[0]]) - 1., np.max([B[0], A[0]]) + 1.])
    ax[0].set_ylim([np.min([B[1], A[1]]) - 1., np.max([B[1], A[1]]) + 1.])
    # title = f"Zermelo solution: {len(X) / 4} days"
    title = f"Zermelo solution: {len(X)*dt/(3600*24)} days, direct flight: {len(X_d)*dt/(3600*24)} days"
    ax[0].set_title(title)

if __name__ == "__main__":
    # Example usage of the Zermelo solution
    # global B
    B = np.array([-65.0, 19.5])  # end point
    A = np.array([-64.3, 18.5])  # start point

    w = 0.25  # speed of glider in m/s
    dt = 3600 * 6  # time of dives in seconds (6 hours)
    dalpha = np.pi / 9  # angle resolution of the heading in radians (10 degrees)
    dl = 0.3 * w * dt  # horizontal resolution of the solution line

    __main__(A, B, w, dt, dalpha, dl)
