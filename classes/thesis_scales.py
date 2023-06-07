import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter
from numpy import ma


class LambertCylindricalLatitudeScale(mscale.ScaleBase):
    """# w    ww  .   d   e m  o  2  s  .   c  o m
    Scales data in range -pi/2 to pi/2 (-90 to 90 degrees) using
    the system used to scale latitudes in a LambertCylindrical__ projection.


    """

    # The scale class must have a member name that defines the string used
    # to select the scale.  For example,
    # gca().set_yscale("lambert_cylindrical")
    # would
    # be used to select this scale.
    name = "lambert_cylindrical"

    def __init__(self, axis, *, thresh=90, **kwargs) -> None:  # type: ignore
        """
        Any keyword arguments passed to set_xscale and set_yscale will
        be passed along to the scale's constructor.

        thresh: The degree above which to crop the data.
        """
        super().__init__(axis)
        if thresh > 90:
            raise ValueError("thresh must be less than 90")
        self.thresh = thresh

    def get_transform(self):  # type: ignore
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The LambertCylindricalLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.LambertCylindricalLatitudeTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):  # type: ignore
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in :mod:`.ticker`.

        In our case, the LambertCylindrical example uses a fixed locator from
        -90 to 90
        degrees and a custom formatter to convert the radians to degrees and
        put a degree symbol after the value.
        """
        fmt = FuncFormatter(lambda x, pos=None: f"{x:.0f}\N{DEGREE SIGN}")
        axis.set(
            major_locator=FixedLocator(
                [-90, -45, -30, -15, 0, 15, 30, 45, 90]
            ),
            major_formatter=fmt,
            minor_formatter=fmt,
        )

    def limit_range_for_scale(self, vmin, vmax, minpos):  # type: ignore
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of LambertCylindrical, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin, -self.thresh), min(vmax, self.thresh)

    class LambertCylindricalLatitudeTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # input_dims and output_dims specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = output_dims = 1

        def __init__(self, thresh):  # type: ignore
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):  # type: ignore
            """
            This transform takes a numpy array and returns a transformed copy.
            Since the range of the LambertCylindrical scale is limited by the
            user-specified threshold, the input array must be masked to
            contain only valid values.  Matplotlib will handle masked arrays
            and remove the out-of-range data from the plot.  However, the
            returned array *must* have the same shape as the input array, since
            these values need to remain synchronized with values in the other
            dimension.
            """
            masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)
            if masked.mask.any():
                return ma.sin(masked * np.pi / 180)
            else:
                return np.sin(a * np.pi / 180)

        def inverted(self):  # type: ignore
            """
            Override this method so Matplotlib knows how to get the
            inverse transform for this transform.
            """
            return LambertCylindricalLatitudeScale.InvertedLambertCylindricalLatitudeTransform(  # noqa
                self.thresh
            )

    class InvertedLambertCylindricalLatitudeTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, thresh):  # type: ignore
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):  # type: ignore
            return np.sinh(a)

        def inverted(self):  # type: ignore
            return LambertCylindricalLatitudeScale.LambertCylindricalLatitudeTransform(  # noqa
                self.thresh
            )


# Now that the Scale class has been defined, it must be registered so
# that Matplotlib can find it.
mscale.register_scale(LambertCylindricalLatitudeScale)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.linspace(-180.0, 180.0, 50)
    s = np.linspace(-90, 90, 50)

    plt.plot(t, s, "-", lw=2)
    plt.gca().set_yscale("lambert_cylindrical")
    plt.ylim(-90, 90)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("LambertCylindrical projection")
    plt.grid(True)

    plt.show()
