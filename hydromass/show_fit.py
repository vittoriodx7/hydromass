import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_likelihood', 'plot_likelihood_sb', 'plot_likelihood_kt', 'plot_likelihood_metal']


def plot_likelihood(Mhyd, trace, dict_utils, fit_SB = False, fit_kT = False, fit_Z_Ab = False, fit_bkg = False, outdir = None):
    '''
    CLAUDE: Plot the model vs observed profiles of every likelihood included in the fit, saving them as likelihood_*.pdf in outdir (default Mhyd.dir).
    '''
    if outdir is None:
        outdir = Mhyd.dir
    if fit_SB:
        plot_likelihood_sb(Mhyd, trace, dict_utils, fit_bkg = fit_bkg, outfile = os.path.join(outdir, 'likelihood_sb.pdf'))
    if fit_kT:
        plot_likelihood_kt(Mhyd, trace, dict_utils, outfile = os.path.join(outdir, 'likelihood_kt.pdf'))
    if fit_Z_Ab:
        plot_likelihood_metal(Mhyd, trace, dict_utils, outfile = os.path.join(outdir, 'likelihood_metal.pdf'))


def _posterior_band(trace, name, index = None):
    '''
    CLAUDE: Median and 68.3% interval of a model profile stored in the trace, computed over all chains and draws.
    '''
    samples = trace.posterior[name].values
    samples = samples.reshape(-1, samples.shape[-1])
    if index is not None:
        samples = samples[:, index]
    return np.percentile(samples, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis = 0)


def _fit_mask(valid, npt):
    '''
    CLAUDE: Boolean mask of the points included in the fit, from the np.where output stored in dict_utils.
    '''
    mask = np.zeros(npt, dtype = bool)
    mask[valid] = True
    return mask


def _new_figure(title):
    '''
    CLAUDE: Create a figure with the model vs observed panel and the residual panel below it.
    '''
    fig, (ax, ax_res) = plt.subplots(2, 1, sharex = True, figsize = (10, 8), gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0})
    ax.set_xscale('log')
    ax.set_title(title)
    ax_res.axhline(0, color = 'k', ls = '--', lw = 1)
    ax_res.set_xlabel('Radius [arcmin]')
    ax_res.set_ylabel('(obs - model) / $\\sigma$')
    return fig, ax, ax_res


def _plot_model_vs_obs(ax, ax_res, data, obs, err, band, mask, color, label):
    '''
    CLAUDE: Plot the model band over all the bins, the points included in the fit in color and the excluded ones in gray.
    '''
    x = data.rref_x_am
    xerr = np.vstack([x - data.rin_x_am, data.rout_x_am - x])
    med, lo, hi = band
    res = (obs - med) / err

    ax.fill_between(x, lo, hi, color = color, alpha = 0.3, lw = 0)
    ax.plot(x, med, color = color, label = f'{label} model')
    ax.errorbar(x[mask], obs[mask], xerr = xerr[:, mask], yerr = err[mask], fmt = 'o', color = color, label = f'{label} fitted')
    ax_res.errorbar(x[mask], res[mask], yerr = 1, fmt = 'o', color = color)
    if np.any(~mask):
        ax.errorbar(x[~mask], obs[~mask], xerr = xerr[:, ~mask], yerr = err[~mask], fmt = 'o', mfc = 'none', color = 'gray', label = f'{label} excluded')
        ax_res.errorbar(x[~mask], res[~mask], yerr = 1, fmt = 'o', mfc = 'none', color = 'gray')


def _save_figure(fig, ax, outfile):
    '''
    CLAUDE: Add the legend and save the figure to outfile, if given.
    '''
    ax.legend()
    if outfile is not None:
        fig.savefig(outfile)
        plt.close(fig)


def plot_likelihood_sb(Mhyd, trace, dict_utils, fit_bkg = False, outfile = None):
    '''
    CLAUDE: Plot the surface brightness (or counts, if fit_bkg) model from the trace against the observed profiles of :func:`likelihood_sb`.
    '''
    fig, ax, ax_res = _new_figure('Surface brightness')
    for i, (this_sb_data, valid_sb) in enumerate(zip(Mhyd.sb_data, dict_utils['valid_sb_list'])):
        if fit_bkg:
            band = _posterior_band(trace, 'cts_model')
            obs = this_sb_data.counts
            err = np.sqrt(np.maximum(this_sb_data.counts, 1.))
        else:
            band = _posterior_band(trace, f'sb_model_{i}')
            obs = this_sb_data.sb
            err = this_sb_data.esb
        mask = _fit_mask(valid_sb, len(obs))
        _plot_model_vs_obs(ax, ax_res, this_sb_data, obs, err, band, mask, f'C{i}', f'SB {i}')
    ax.set_yscale('log')
    ax.set_ylabel('Counts' if fit_bkg else 'Surface brightness')
    _save_figure(fig, ax, outfile)
    return fig


def plot_likelihood_kt(Mhyd, trace, dict_utils, outfile = None):
    '''
    CLAUDE: Plot the projected temperature model from the trace against the observed profiles of :func:`likelihood_kt`.
    '''
    fig, ax, ax_res = _new_figure('Temperature')
    for i, (this_spec_data, valid_spec) in enumerate(zip(Mhyd.spec_data, dict_utils['valid_spec_list'])):
        band = _posterior_band(trace, f'tproj_{i}')
        mask = _fit_mask(valid_spec, len(this_spec_data.temp_x))
        _plot_model_vs_obs(ax, ax_res, this_spec_data, this_spec_data.temp_x, this_spec_data.errt_x, band, mask, f'C{i}', f'kT {i}')
    ax.set_ylabel('kT [keV]')
    _save_figure(fig, ax, outfile)
    return fig


def plot_likelihood_metal(Mhyd, trace, dict_utils, outfile = None):
    '''
    CLAUDE: Plot the metal abundance model from the trace against the observed profiles of :func:`likelihood_metal`.
    '''
    fig, ax, ax_res = _new_figure('Metal abundance')
    for i, (this_spec_data, index_x, valid_spec) in enumerate(zip(Mhyd.spec_data, dict_utils['index_kt_list'], dict_utils['valid_spec_list'])):
        band = _posterior_band(trace, 'zfe_model', index = index_x)
        mask = _fit_mask(valid_spec, len(this_spec_data.zfe))
        _plot_model_vs_obs(ax, ax_res, this_spec_data, this_spec_data.zfe, this_spec_data.errt_zfe, band, mask, f'C{i}', f'Z {i}')
    ax.set_ylabel('Z$_{Fe}$ [Z$_\\odot$]')
    _save_figure(fig, ax, outfile)
    return fig
