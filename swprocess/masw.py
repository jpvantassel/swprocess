# This file is part of swprocess, a Python package for surface wave processing.
# Copyright (C) 2020 Joseph P. Vantassel (joseph.p.vantassel@gmail.com)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Masw class definition."""

import logging

import numpy as np

from .maswworkflows import MaswWorkflowRegistry

logger = logging.getLogger("swprocess.masw")


class Masw():
    """Customizable Multichannel Analysis of Surface Waves workflow.

    Convenient customer-facing interface for implementing different
    and extensible MASW processing workflows.

    """

    @staticmethod
    def run(fnames, settings, map_x=lambda x: x, map_y=lambda y: y):
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
        settings_fname : str
            JSON settings file detailing how MASW should be performed.
            See `meth: Masw.create_settings_file()` for more
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
        # TODO (jpv): Rewrite docstring

        # # Load settings.
        # with open(settings_fname, "r") as f:
        #     settings = json.load(f)

        # Acquire Workflow (class) from registry.
        selected_workflow = settings["workflow"]
        logger.info(f"selected workflow is {selected_workflow}")
        Workflow = MaswWorkflowRegistry.create_class(selected_workflow)

        # Define workflow (instance) from Workflow (class).
        workflow = Workflow(fnames=fnames, settings=settings,
                            map_x=map_x, map_y=map_y)

        # Run and return.
        return workflow.run()

    @staticmethod
    def create_settings_dict(workflow="time-domain",
                             trim=False, trim_begin=0.0, trim_end=1.0,
                             mute=False, method="interactive",
                             window_kwargs=None, pad=False, df=1.0,
                             transform="fdbf", fmin=5, fmax=100, vmin=100,
                             vmax=1000, nvel=200, vspace="linear",
                             weighting="sqrt", steering="cylindrical",
                             snr=False, noise_begin=-0.5, noise_end=0.0,
                             signal_begin=0.0, signal_end=0.5,
                             pad_snr=True, df_snr=1.0, min_offset=0,
                             max_offset=np.inf):
        """Create settings `dict` using function arguments.

        See :meth:`Masw.create_settings_file` for details.

        """
        return {"workflow": str(workflow),
                "pre-processing": {
                    "trim": {
                        "apply": bool(trim),
                        "begin": float(trim_begin),
                        "end": float(trim_end)
                    },
                    "mute": {
                        "apply": bool(mute),
                        "method": str(method),
                        "window_kwargs": {} if window_kwargs is None else dict(window_kwargs)
                    },
                    "pad": {
                        "apply": bool(pad),
                        "df": float(df)
                    },
                    "offsets": {
                        "min": float(min_offset),
                        "max": float(max_offset),
                    }
        },
            "processing": {
                    "transform": str(transform),
                    "fmin": float(fmin),
                    "fmax": float(fmax),
                    "vmin": float(vmin),
                    "vmax": float(vmax),
                    "nvel": int(nvel),
                    "vspace": str(vspace),
                    "fdbf-specific": {
                        "weighting": str(weighting),
                        "steering": str(steering)
                    }
        },
            "signal-to-noise": {
                    "perform": bool(snr),
                    "noise": {
                        "begin": float(noise_begin),
                        "end": float(noise_end)
                    },
                    "signal": {
                        "begin": float(signal_begin),
                        "end": float(signal_end)
                    },
                    "pad": {
                        "apply": bool(pad_snr),
                        "df": float(df_snr)
                    }
        },
        }

    # @staticmethod
    # def create_settings_file(fname, workflow="time-domain",
    #                          trim=False, trim_begin=0.0, trim_end=1.0,
    #                          mute=False, method="interactive",
    #                          window_kwargs=None, pad=False, df=1.0,
    #                          transform="fdbf", fmin=5, fmax=100, vmin=100,
    #                          vmax=1000, nvel=200, vspace="linear",
    #                          weighting="sqrt", steering="cylindrical",
    #                          snr=False, noise_begin=-0.5, noise_end=0.0,
    #                          signal_begin=0.0, signal_end=0.5,
    #                          pad_snr=True, df_snr=1.0):
    #     """Create settings file using function arguments.

    #     Parameters
    #     ----------
    #     fname : str
    #         Name of file where settings will be saved. May include a
    #         relative or the full path.
    #     workflow : {'time-domain', 'frequency-domain', 'single'}, optional
    #         Name of MASW processing workflow, `default is
    #         'time-domain'`.
    #     trim : bool, optional
    #         Denote whether time records are to be trimmed, default is
    #         `False`.
    #     trim_begin, trim_end : float, optional
    #         If `trim` is `True`, these define the trimming start and
    #         end time in seconds.
    #     mute : bool, optional
    #         Denote whether time-domain muting is to be performed,
    #         default is `False`.
    #     method : {'interactive'}, optional
    #         If `mute` is `True`, select the method for performing
    #         time-domain muting.
    #     window_kwargs : dict, optional
    #         If `mute` is `True`, describe the shape of the mute mask,
    #         see `scipy.singal.windows.tukey <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
    #         for available options.
    #     pad : bool, optional
    #         Perform time-domain padding to attain a specific frequency
    #         step `df`, default is `False`.
    #     df : float, optional
    #         Desired frequency step in Hz if `pad=True`, default is 1 Hz.
    #     transform : {"fdbf", "phaseshift", "slantstack"}, optional
    #         Multichannel transformation, default is `fdbf`.
    #     fmin, fmax : float, optional
    #         Minimum and maximum processing frequencies, default is `5`
    #         and `100`, respectively.
    #     vmin, vmax : float, optional
    #         Minimum and maximum processing velocity in m/s, default is
    #         `100` and `1000`, respectively.
    #     nvel : int, optional
    #         Number of velocity steps, default is `200`.
    #     vpsace : {'linear', 'log'}, optional
    #         Select the whether the `nvel` trial velocities are selected
    #         in `linear` or `log` space, default is `linear`.
    #     weighting : {'sqrt', 'invamp', 'none'}, optional
    #         If `transform='fdbf', then select weighting, default is
    #         `'sqrt'`.
    #     steering ; {'cylindrical', 'plane'}, optional
    #         If `transform='fdbf', then select steering, default is
    #         `'cylindrical'`.
    #     snr : bool, optional
    #         Determine whether signal-to-noise ratio calculation should
    #         be performed, default is `False`.
    #     noise_begin, noise_end : float, optional
    #         If `snr=True`, select noise window start and end time
    #         respectively, default is `-0.5`, `0` seconds respectively.
    #     signal_begin, signal_end : float, optional
    #         If `snr=True`, select signal window start and end time
    #         respectively, default is `0`, `0.5` seconds respectively.
    #     pad_snr : bool, optional
    #         If `snr=True`, select whether singal-to-noise ratio windows
    #         should be padded. If singal and noise windows are of
    #         different lengths, this must be `True`, default is `True`.
    #     df_snr : float, optional
    #         If `snr=True` and `pad_snr=True`, set the desired frequency
    #         domain spacing, default is 1 Hz.

    #     """
    #     settings = Masw.create_settings_dict(workflow=workflow,
    #                                          trim=trim,
    #                                          trim_begin=trim_begin,
    #                                          trim_end=trim_end,
    #                                          mute=mute, method=method,
    #                                          window_kwargs=window_kwargs,
    #                                          pad=pad, df=df, transform=transform,
    #                                          fmin=fmin, fmax=fmax, vmin=vmin,
    #                                          vmax=vmax, nvel=nvel, vspace=vspace,
    #                                          weighting=weighting, steering=steering,
    #                                          snr=snr, noise_begin=noise_begin,
    #                                          noise_end=noise_end,
    #                                          signal_begin=signal_begin,
    #                                          signal_end=signal_end,
    #                                          pad_snr=pad_snr, df_snr=df_snr)
    #     with open(fname, "w") as f:
    #         json.dump(settings, f)


class MaswXcorr(Masw):

    @staticmethod
    def run(fnames_rec, fnames_src, src_channel, settings,
            map_x=lambda x: x, map_y=lambda y: y):

        # Acquire Workflow (class) from registry.
        if settings["workflow"] != "time-domain-xcorr":
            raise NotImplementedError

        selected_workflow = settings["workflow"]
        logger.info(f"selected workflow is {selected_workflow}")
        Workflow = MaswWorkflowRegistry.create_class(selected_workflow)

        # Define workflow (instance) from Workflow (class).
        workflow = Workflow(fnames_rec=fnames_rec,
                            fnames_src=fnames_src,
                            src_channel=src_channel,
                            settings=settings,
                            map_x=map_x,
                            map_y=map_y)

        # Run and return.
        return workflow.run()
