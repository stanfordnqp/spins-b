"""Functions to plot data from the log files output during optimization."""

import math
from typing import List, Optional

from matplotlib.backends import backend_pdf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

from spins.invdes.problem_graph.log_tools import loader
from spins.invdes.problem_graph.log_tools import monitor_spec


def plot_scalar_monitors(
        monitor_description_list: monitor_spec.MonitorDescriptionList,
        log_df: pd.DataFrame,
        event_name: str = "optimizing",
        same_plt: bool = False,
        pdf_obj: Optional[backend_pdf.PdfPages] = None,
        filename: Optional[str] = None,
        show: bool = True):
    """Plots scalar monitors versus iteration.

    The data is joined across transformations, where monitor data is found for
    the given monitor and event name. Vertical black dashed lines indicate
    transformation changes.

    Args:
        monitor_descriptions_list: `MonitorDescriptionList` object for scalar
            monitors to plot.
        log_df: Pandas dataframe containing all spins optimization output
            information.
        event_name: String specifying the event - default is "optimizing" for
            iteration data.
        same_plt: Boolean value indicating whether the monitors should be
            plotted on the same plot or separate plots.
        filename: String containing filename to save the generated figures to.
        pdf_obj: PdfPages object to which the figures generated in this function
            are saved in a multipage pdf. This should be used to generate
            multipage pdf of a set of results.
        show: Boolean indicating whether or not the generated figures should be
            displayed to screen.
    """
    # Get figure and axis objects to plot the data on.
    if same_plt:
        num_plts = 1
        num_rows = 1
        num_cols = 1
    else:
        num_plts = len(monitor_description_list.monitor_list)
        num_rows = 3
        num_cols = 2

    fig_list = _create_figs(num_plts, num_rows, num_cols)
    axes_list = []
    for fig in fig_list:
        axes_list += fig.axes

    for plot_num, monitor_description in enumerate(
            monitor_description_list.monitor_list):
        # Get all the transformation data for the monitor name and join it.
        # Default to "real" in case we receive complex data.
        # Currently this is used as a hack since objective function values
        # can actually "become" complex.
        scalar_op = monitor_description.scalar_operation
        if not scalar_op:
            scalar_op = "real"

        plot_data = loader.get_joined_scalar_monitors(
            log_df, monitor_description.monitor_names, event_name, scalar_op)

        # Choose axis object to plot on.
        if same_plt:
            axs = axes_list[0]
        else:
            axs = axes_list[plot_num]

        axs.plot(
            plot_data.iterations,
            plot_data.data,
            label=monitor_description.joiner_id)
        for change in plot_data.transformation_changes:
            axs.axvline(x=change, color="k", linestyle="--")
        if same_plt:
            axs.legend()
        else:
            if not plot_data.data.size:
                axs.set_title("{name}:\nNo iteration data found.".format(
                    name=monitor_description.joiner_id))
            else:
                axs.set_title("{name}:\nFinal value: {value:1.4E}".format(
                    name=monitor_description.joiner_id,
                    value=plot_data.data[-1]))
        axs.set_xlabel("Iteration")

    # Save generated figures to multipage pdf object.
    if pdf_obj is not None:
        for fig in fig_list:
            pdf_obj.savefig(fig)

    # Save generated figures to multipage pdf object specified by filename.
    if filename is not None:
        with backend_pdf.PdfPages(filename) as pdf:
            for fig in fig_list:
                pdf.savefig(fig)
    if show:
        plt.show()


def plot_field_data(
        monitor_description_list: monitor_spec.MonitorDescriptionList,
        log_df: pd.DataFrame,
        event_name: Optional[str] = None,
        transformation_name: Optional[str] = None,
        iteration: Optional[int] = None,
        pdf_obj: Optional[backend_pdf.PdfPages] = None,
        filename: Optional[str] = None,
        show: bool = True) -> None:
    """Plots planar monitor data.

    Args:
        monitor_description_list: `MonitorDescriptionList` object for planar
            monitors to plot.
        log_df: Pandas dataframe containing all spins optimization output
            information.
        event_name: String specifying the event - default is "optimizing" for
            iteration data.
        transformation_name: String specifying the transformation of interest.
            If `None` is given, data for the last transformation performed is
            plotted.
        iteration: The iteration of interest. If `None` is given, data
            for the last iteration available is plotted.
        component: String specifying x, y, or z component. If `None`, the
            magnitude is plotted.
        pdf_obj: PdfPages object to which the figures generated in this function
            are saved in a multipage pdf. This is useful to save differen
            results into a single pdf.
        filename: String containing the filename to save the generated figures
            to.
        show: Boolean indicating whether or not to display generated figures to
            screen.
    """
    # Get figure and axis objects to plot the data on.
    num_plts = len(monitor_description_list.monitor_list)
    fig_list = _create_figs(num_plts, num_rows=2, num_cols=2)
    axes_list = []
    for fig in fig_list:
        axes_list += fig.axes

    for plt_ind in range(num_plts):
        axs = axes_list[plt_ind]

        # Get the magnitude data.
        monitor_description = monitor_description_list.monitor_list[plt_ind]
        field_dat = loader.get_single_monitor_data(
            log_df,
            monitor_description.monitor_names,
            transformation_name=transformation_name,
            iteration=iteration,
            event_name=event_name)
        if field_dat:
            field = loader.process_field(
                field_dat,
                vector_operation=monitor_description.vector_operation,
                scalar_operation=monitor_description.scalar_operation)
            field = np.squeeze(np.array(field.T))

            # Make sure field is plottable as an image.
            if len(field.shape) != 2:
                raise ValueError(
                    "Plotted field data must be 2D, but {} has dimensions {}".
                    format(monitor_description.joiner_id, len(field.shape)))

            im_plt = axs.imshow(field, origin="lower")
            axs.set_title(monitor_description.joiner_id)
            cax = make_axes_locatable(axs).append_axes(
                "right", size="5%", pad=0.15)
            plt.colorbar(im_plt, ax=axs, cax=cax, format="%.2e")
        else:
            axs.set_title("{name}:\nNo field data found.".format(
                name=monitor_description.joiner_id))
    # Save generated figures to multipage pdf object.
    if pdf_obj is not None:
        for fig in fig_list:
            pdf_obj.savefig(fig)

    # Save generated figures to multipage pdf object specified by filename.
    if filename is not None:
        with backend_pdf.PdfPages(filename) as pdf:
            for fig in fig_list:
                pdf.savefig(fig)

    if show:
        plt.show()


def _create_figs(num_plts: int, num_rows: int = 1, num_cols: int = 1) -> List:
    """Creates a list of figures ont which to plot data.

    Creates a series of figures with `num_rows` rows and `num_cols` columns
    for total of `num_plts` subplots. This is used so that each figure can
    be fit onto a single PDF page.

    Args:
        num_plts: Total number of plots to make.
        num_rows: Number of rows for desired grid of subplots.
        num_cols: Number of columns for desired grid of subplots.
    """

    num_figs = math.ceil(num_plts / (num_rows * num_cols))
    extras = num_plts % (num_rows * num_cols)
    figs_list = []
    for fig_ind in range(num_figs):
        fig, _ = plt.subplots(
            num_rows, num_cols, figsize=(10, 6), tight_layout=True)
        figs_list.append(fig)

    # Remove extra axes if necessary.
    if extras:
        for extras_ind in range((num_rows * num_cols) - extras):
            figs_list[-1].delaxes(figs_list[-1].axes[-1])

    return figs_list


def plot_overlaps(log_df: pd.DataFrame,
                  compute_complex: str = "power",
                  same_plt: bool = True,
                  pdf_obj: Optional[backend_pdf.PdfPages] = None,
                  filename: Optional[str] = None,
                  show: bool = True):
    """Plots overlap monitors versus iteration.

    Data is joined across transformations, where monitor data is found for the
    given monitor and event name. Vertical black dashed lines indicate
    transformation changes.

    Args:
        log_df: Pandas dataframe containing all spins optimization output information.
        compute_complex: String indicating what operation to perform on overlap monitor outputs.
            Possibilities include "power","phase","real",and "imag".
        same_plt: Boolean value indicating whether the monitors should be plotted
            on the same plot or separate plots.
        filename: String containing filename to save the generated figures to.
        pdf_obj: PdfPages object to which the figures generated in this function
            are saved in a multipage pdf. This should be used to generate
            multipage pdf of a set of results.
        show: Boolean indicating whether or not the generated figures should be displayed to
            screen.
    """
    # Retrieve names of all overlap monitors.
    overlap_names = loader.get_overlap_monitor_names(log_df)

    # Plot overlap monitors.
    plot_scalar_monitors(
        monitor_name_list=overlap_names,
        log_df=log_df,
        same_plt=same_plt,
        compute_complex=compute_complex,
        pdf_obj=pdf_obj,
        filename=filename,
        show=show)


def plot_monitor_data(log_df: pd.DataFrame,
                      monitor_descriptions: monitor_spec.MonitorDescriptionList,
                      filename: Optional[str] = None,
                      show: bool = True) -> None:
    """Plots all monitor data.

    Specifically, plots all available scalar monitor data versus iteration and
    all planar monitor magnitude data from the end of the optimization.
    If a filename is given, saves all generated figures to multipage pdf.

    Args:
        log_df: Pandas dataframe containing all spins optimization output
            information.
        filename: String giving the filename where all generated figures are
            saved as a multipage pdf.
        show: Whether the plots should be displayed to the screen.
    """
    monitor_list = loader.get_monitors_by_type(log_df, monitor_descriptions)
    if filename is not None:
        with backend_pdf.PdfPages(filename) as pdf:
            plot_scalar_monitors(
                monitor_list.scalars, log_df, pdf_obj=pdf, show=False)
            plot_field_data(
                monitor_list.planars, log_df, pdf_obj=pdf, show=False)
    else:
        plot_scalar_monitors(monitor_list.scalars, log_df, show=False)
        plot_field_data(monitor_list.planars, log_df, show=False)

    if show:
        plt.show()
    else:
        plt.close()
