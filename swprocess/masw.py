"""Masw class definition."""

import logging
import json

from .workflows import MaswWorkflowRegistry

logger = logging.getLogger(__name__)


class Masw():
    """Customizable Multichannel Analysis of Surface Waves workflow.

    Convenient customer-facing interface for implementing different
    and extensible MASW processing workflows.

    """

    @staticmethod
    def run(fnames, settings_fname, map_x=lambda x: x, map_y=lambda y: y):
        """Run an MASW workflow from SU or SEGY files.

        Create an instance of an `Masw` object for a specific
        `Masw` workflow. Note that each file should contain
        multiple traces where each trace corresponds to a
        single receiver. The header information for these files must
        be correct and readable. Currently supported file types are
        SEGY and SU.

        Parameters
        ----------
        fnames : str or iterable of str
            File name or iterable of file names.
        settings : str
            JSON settings file detailing how MASW should be performed.
            See `meth: Masw.example_settings_file()` for more
            information.
        map_x, map_y : function, optional
            Functions to convert the x and y coordinates of source and
            receiver information, default is no transformation. Useful
            for converting between coordinate systems.

        Returns
        -------
        AbstractTransform-like
            Initialized subclass (i.e., child) of `AbstractTransform`.

        Raises
        ------
        TypeError
            If `fnames` is not of type `str` or `iterable`.

        """
        # Load settings
        with open(settings_fname, "r") as f:
            settings = json.load(f)

        Workflow = MaswWorkflowRegistry.create_class(settings["workflow"])
        workflow = Workflow(fnames=fnames, settings=settings, map_x=map_x,
                            map_y=map_y)
        return workflow.run()

    @classmethod
    def example_settings_file(cls, fname, workflow="time-domain",
                              trim=False, start_time=0.0, end_time=1.0,
                              mute=False, method="interactive",
                              window_kwargs=None, pad=False, df=1.,
                              transform="fdbf", fmin=5, fmax=100, vmin=100,
                              vmax=400, nvel=100, vspace="linear",
                              weighting="sqrt", steering="cylindrical",
                              snr=False, noise_begin=-0.5, noise_end=0.0,
                              signal_begin=0.0, signal_end=0.5,
                              pad_snr=True, df_snr=1.0):
        settings = {"workflow": workflow,
                    "pre-processing": {
                        "trim": {
                            "apply": trim,
                            "begin": start_time,
                            "end": end_time
                        },
                        "mute": {
                            "apply": mute,
                            "method": method,
                            "window_kwargs": window_kwargs if window_kwargs is not None else {}
                        },
                        "pad": {
                            "apply": pad,
                            "df": df
                        }
                    },
                    "processing": {
                        "transform": transform,
                        "fmin": fmin,
                        "fmax": fmax,
                        "vmin": vmin,
                        "vmax": vmax,
                        "nvel": nvel,
                        "vspace": vspace,
                        "fdbf-specfic": {
                            "weighting": weighting,
                            "steering": steering
                        }
                    },
                    "signal-to-noise": {
                        "perform": snr,
                        "noise": {
                            "begin": noise_begin,
                            "end": noise_end
                        },
                        "signal": {
                            "begin": signal_begin,
                            "end": signal_end
                        },
                        "pad": {
                            "apply": pad_snr,
                            "df": df_snr
                        }
                    }
                    }
        with open(fname, "w") as f:
            json.dump(settings, f)