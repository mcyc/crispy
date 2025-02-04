import numpy as np
import pandas as pd
import astropy.io.fits as fits
import time
from crispy.scms import find_ridge  # Keeps compatibility with existing CRISPy functions
from skimage import morphology  # Needed for small object filtering


class RidgeFinder:
    """
    A high-level interface for CRISPy's SCMS-based ridge detection.

    This class allows for flexible data input (images, tables), automated masking,
    and optimized SCMS ridge finding while maintaining backward compatibility with CRISPy.
    """

    def __init__(self, data=None, h=1.0, eps=1e-2, max_iter=1000, max_time=None, ncpu=None,
                 mask_method=None, walker_strategy=None, backend="cpu"):
        """
        Initialize RidgeFinder with optional data and SCMS parameters.

        Parameters
        ----------
        data : str, np.ndarray, or pd.DataFrame, optional
            File path to an image/table OR pre-loaded NumPy array or DataFrame.

        h : float, optional
            Smoothing bandwidth for SCMS (default: 1.0).

        eps : float, optional
            Convergence threshold for SCMS (default: 1e-2).

        max_iter : int, optional
            Maximum number of SCMS iterations (default: 1000).

        max_time : float, optional
            Maximum runtime for SCMS in **hours** (default: None, no time limit).

        ncpu : int, optional
            Number of CPUs for parallel computation (default: None, uses all available CPUs).

        mask_method : str, optional
            Strategy for generating a selection mask (default: None, automatic selection).

        walker_strategy : str, optional
            Strategy for placing walkers (default: None, automatic selection).

        backend : {"cpu", "gpu"}, optional
            Processing backend (default: "cpu"). Future support for GPU (CuPy, JAX).
        """
        # Store SCMS parameters in a unified config dictionary
        self.config = {
            "h": h,
            "eps": eps,
            "max_iter": max_iter,
            "max_time": max_time * 3600 if max_time else None,  # Convert hours to seconds
            "ncpu": ncpu
        }

        # Store backend configuration
        self.backend = backend

        # Data storage attributes
        self.data = None
        self.data_points = None
        self.image_mask = None
        self.walker_mask = None
        self.active_walkers = None
        self.converged_walkers = None

        # Strategy attributes (modular, allows dynamic change later)
        self.mask_method = mask_method
        self.walker_strategy = walker_strategy

        # Load data if provided
        if data is not None:
            self.load_data(data)

    def load_data(self, data):
        """
        Loads data from a file path or a pre-loaded NumPy/Pandas object.

        Parameters
        ----------
        data : str, np.ndarray, or pd.DataFrame
            File path to an image/table OR pre-loaded NumPy array or DataFrame.

        Raises
        ------
        ValueError
            If the input data type is unsupported.
        """
        if isinstance(data, str):
            # Auto-detect and load based on file extension
            if data.endswith(".fits"):
                self.data = fits.getdata(data)
            elif data.endswith(".csv"):
                self.data = pd.read_csv(data)
            elif data.endswith(".parquet"):
                self.data = pd.read_parquet(data)
            else:
                raise ValueError(f"Unsupported file format: {data}")

        elif isinstance(data, np.ndarray):
            self.data = data  # Directly store NumPy array

        elif isinstance(data, pd.DataFrame):
            self.data = data  # Directly store Pandas DataFrame

        else:
            raise ValueError("Data must be a file path, NumPy array, or Pandas DataFrame.")

        print(f"✔ Data successfully loaded: {type(self.data).__name__}")

    def print_config(self):
        """
        Print the current SCMS configuration.
        """
        print("\nRidgeFinder Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")


    def convert_to_points(self, min_size=9):
        """
        Converts the loaded data (image or structured table) into SCMS-compatible points.
        Determines the data type and calls the appropriate conversion function.

        Parameters
        ----------
        min_size : int, optional
            Minimum size (in pixels) for structures to be retained (default: 9).
            Smaller structures are removed to suppress noise.

        Raises
        ------
        ValueError
            If no data is loaded or an unsupported data format is encountered.
        """
        if self.data is None:
            raise ValueError("No data loaded. Use `load_data()` first.")

        if isinstance(self.data, np.ndarray):
            self.data_points = self.image_to_points(min_size)
        elif isinstance(self.data, pd.DataFrame):
            self.data_points = self.dataframe_to_points()
        else:
            raise ValueError(
                "Unsupported data format. Must be an image (NumPy array) or structured table (Pandas DataFrame).")

        print(f"✔ Data successfully converted: {len(self.data_points['X'])} points extracted.")


    def image_to_points(self, min_size=9):
        """
        Converts an image (NumPy array) into SCMS-compatible points.

        Extracts pixel coordinates and intensities, applies masking, and filters small objects.

        Parameters
        ----------
        min_size : int, optional
            Minimum size (in pixels) for structures to be retained (default: 9).

        Returns
        -------
        dict
            Dictionary containing:
            - `"X"`: Extracted pixel coordinates as an (N, D, 1) NumPy array.
            - `"weights"`: Corresponding pixel intensities.
        """
        image = self.data
        im_shape = image.shape
        D = len(im_shape)  # Dimensionality (2D or 3D)
        indices = np.indices(im_shape)  # Get pixel coordinates

        # Apply existing image mask if available, otherwise default to a percentile-based mask
        mask = self.image_mask if self.image_mask is not None else (image > np.percentile(image, 10))

        # Remove small objects if needed
        mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=1)

        # Extract valid pixel coordinates & intensities
        X = np.array([i[mask] for i in indices])  # Coordinates
        X = X[np.newaxis, :].swapaxes(0, -1)  # Shape (N, D, 1)

        weights = image[mask]  # Pixel intensities

        return {"X": X, "weights": weights}


    def dataframe_to_points(self):
        """
        Converts a structured Pandas DataFrame into SCMS-compatible points.

        Extracts the necessary columns (x, y, [optional z, weights]) and converts them to NumPy format.

        Returns
        -------
        dict
            Dictionary containing:
            - `"X"`: Extracted structured data coordinates as an (N, D, 1) NumPy array.
            - `"weights"`: Corresponding weights, defaulting to 1 if missing.

        Raises
        ------
        ValueError
            If required columns ('x', 'y') are missing.
        """
        if not {"x", "y"}.issubset(self.data.columns):  # Ensure mandatory columns exist
            raise ValueError("Structured data must contain at least 'x' and 'y' columns.")

        columns = ["x", "y"] + [col for col in self.data.columns if col not in ["x", "y"]]
        X = self.data[columns].to_numpy().reshape(-1, len(columns), 1)  # Shape (N, D, 1)

        # Use the third column as weights if available, otherwise default to ones
        weights = self.data.iloc[:, 2].to_numpy() if len(columns) > 2 else np.ones(len(X))

        return {"X": X, "weights": weights}
