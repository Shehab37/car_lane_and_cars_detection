import numpy as np


class Line():

    def __init__(self):

        self.fit_px = []
        self.fit_m = None
        self.radius_of_curvature = None
        self.lane_to_camera = None
        self.diffs = np.array([0, 0, 0], dtype=float)
        self.ym_per_pix = 30/720

        # y_eval is where we want to evaluate the fits for the line radius calcuation
        # for us it's at the bottom of the image for us, and because we know
        # the size of our video/images we can just hardcode it
        self.y_eval = 720. * self.ym_per_pix

        # camera position is where the camera is located relative to the image
        # we're assuming it's in the middle
        self.camera_position = 640.

    def add_new_fit(self, new_fit_px, new_fit_m):

        # If this is our first line, then we will have to take it
        self.detected = True
        self.fit_px = new_fit_px
        self.fit_m = new_fit_m
        self.calc_radius()
        return

    def calc_radius(self):
        """
        left_fit and right_fit are assumed to have already been converted to meters
        """
        y_eval = self.y_eval
        fit = self.fit_m

        curve_rad = ((1 + (2*fit[0]*y_eval + fit[1])**2)
                     ** 1.5) / np.absolute(2*fit[0])
        self.radius_of_curvature = curve_rad
        return
