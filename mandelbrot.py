import os

if "LEGATE_MAX_DIM" in os.environ and "LEGATE_MAX_FIELDS" in os.environ:
    import cunumeric as np
    from legate.timing import time as _time

    def time():
        return _time()


else:
    import numpy as np
    import time as t

    def time():
        return t.time_ns() / 1e3


from matplotlib import pyplot as plt
from matplotlib import colors


print(f"Imported NumPy-like package: {np.__name__}")


class Mandelbrot:
    def __init__(
        self,
        xmin: float = -2.0,
        xmax: float = 2.0,
        ymin: float = -1.25,
        ymax: float = 1.25,
        nx: int = 10,
        ny: int = 10,
        maxiter: int = 10,
    ):
        """Initialize the complex function z by linearly 
        discretizing the x and y axes

        Parameters:
        -----------
        xmin, xmax: int, int
            Domain extents along x direction
        ymin, ymax: int, int
            Domain extents along y direction
        nx, ny: int, int
            Resolution of discretization along x and y directions
        maxiter: int
            Number of iterations 
        """

        self._nx = nx
        self._ny = ny
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._maxiter = maxiter
        self._initialized = False

        self._initialize()

    def _initialize(self):
        if not self._initialized:
            r1 = np.linspace(self._xmin, self._xmax, self._nx)
            r2 = np.linspace(self._ymin, self._ymax, self._ny)

            self.z = r1[:, np.newaxis] + 1j * r2[np.newaxis, :]
            self.c = self.z
            self._initialized = True

    def generate_set(self, max_magnitude: float = 2.0):
        "This function was modified from Reference 1"

        self.output = np.zeros(self.c.shape, dtype=np.int32)
        start = time()
        for iter_count in range(self._maxiter):
            self.z = self.z * self.z + self.c

            # a boolean mask that holdsthe status of whether the termination criteria
            # for each element was met (True) or not (False)
            done = np.greater(np.abs(self.z), max_magnitude)

            # for elements where the computations are done, return (0, 0).
            # for other elements, set it to the value of c or z
            self.c = np.where(done, 0 + 0j, self.c)
            self.z = np.where(done, 0 + 0j, self.z)

            # output will hold the number of iterations when the computations are done
            self.output = np.where(done, iter_count, self.output)

        elapsed_time_ms = (time() - start) / 1e3
        self._initialized = False

        return elapsed_time_ms

    def plot(self, figname: str = "fractal.png"):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(
            self.output.T, cmap="gnuplot2", origin="lower", norm=colors.PowerNorm(0.3)
        )
        plt.savefig(figname)


mandelbrot_args = {
    "xmin": -2.0,
    "xmax": 0.5,
    "ymin": -1.25,
    "ymax": 1.25,
    "nx": 6000,
    "ny": 6000,
    "maxiter": 10,
}

nreps = 1
elapsed_time_ms = 0.0
for _ in range(nreps):
    mandelbrot = Mandelbrot(**mandelbrot_args)
    elapsed_time_ms += mandelbrot.generate_set()
    mandelbrot.plot()

elapsed_time_ms /= nreps

print(
    f"The mandelbrot set computation took {elapsed_time_ms} ms for the following input args:"
)
print(f"{mandelbrot_args}")
