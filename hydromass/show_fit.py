import os

import ipdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from scipy import stats
from getdist import MCSamples, plots

from .get_delta import summarize_delta, write_delta_table, DELTA_COLUMNS

__all__ = ['compute_products', 'save_products', 'plot_fits', 'plot_likelihood', 'plot_mass', 'plot_mass_fractions', 'plot_corner', 'plot_thermodynamics']

# Likelihood kinds: (figure title, short label, y label, log y axis)
LIKELIHOOD_KINDS = {
    'sb': ('Surface brightness', 'SB', 'Surface brightness', True),
    'kt': ('Temperature', 'kT', 'kT [keV]', False),
    'metal': ('Metal abundance', 'Z', 'Z$_{Fe}$ [Z$_\\odot$]', False),
}


# ---------------------------------------------------------------------------------------------------------------------
# Products: everything needed for the plots, computed once from the trace of a fit
# ---------------------------------------------------------------------------------------------------------------------

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


def _likelihood_dataset(data, obs, err, band, mask):
    '''
    CLAUDE: Observed profile, model band and fit mask of one dataset, as stored in the products.
    '''
    return dict(x = data.rref_x_am, xlo = data.rin_x_am, xhi = data.rout_x_am, obs = obs, err = err, band = band, mask = mask)


def _corner_inputs(trace, models, dict_utils):
    '''
    CLAUDE: Samples, getdist names, labels, ranges and priors of the free parameters of models (and logp0) for the corner plot.
    '''
    post = trace.posterior.stack(sample = ('chain', 'draw'))

    data, names, labels, ranges, priors = [], [], [], dict(), []
    for model in models:
        for i, par in enumerate(model.parnames):
            if model.fix[i]:
                continue
            # getdist names must be plain identifiers, e.g. 'log M_tot,miss' -> 'log_M_tot_miss'
            name = ''.join(c if c.isalnum() else '_' for c in par)
            data.append(post[par].values)
            names.append(name)
            # Escape characters that getdist (mathtext) would interpret, e.g. '_' as subscript
            labels.append(par.replace(' ', '\\ ').replace('_', '\\_'))
            if model.prior[i] != 'Normal':
                ranges[name] = [float(model.limits[i][0]), float(model.limits[i][1])]
            priors.append((model.prior[i], model.start[i], model.sd[i], model.limits[i][0], model.limits[i][1]))

    if 'logp0' in post:
        # Same prior as in hydromass.likelihoods.kt_from_mass
        P0_est, err_P0_est = dict_utils['P0_est'], dict_utils['err_P0_est']
        lower, upper = np.log(P0_est) - 2 * np.log(10), np.log(P0_est) + 2 * np.log(10)
        data.append(post['logp0'].values)
        names.append('logp0')
        labels.append('\\log P_0')
        ranges['logp0'] = [lower, upper]
        priors.append(('TruncatedNormal', np.log(P0_est), err_P0_est / P0_est, lower, upper))

    samples = np.array(data).T if len(data) > 0 else np.zeros((0, 0))
    return dict(samples = samples, names = names, labels = labels, ranges = ranges, priors = priors)


def compute_products(Mhyd, trace, dict_utils, tag, models_mass, model_mstar = None, mstar = None, model_missing_mass = None,
                     fit_SB = False, fit_kT = False, fit_Z_Ab = False, fit_bkg = False, delta_results = None):
    '''
    CLAUDE: Compute from the trace of a fit everything needed by the plots of this module (likelihood profiles, mass profiles,
    mass fractions, overdensity quantities, corner plot inputs). The result can be plotted alone or together with other fits
    with :func:`plot_fits`, and saved with :func:`save_products`.

    :return: dictionary of products
    '''
    rref_m = dict_utils['rref_m']
    products = dict(tag = tag, rref_m = rref_m, rmin_kpc = dict_utils['rmin_amin'] * Mhyd.amin2kpc, rmax_kpc = dict_utils['rmax_amin'] * Mhyd.amin2kpc,
                    fit_bkg = fit_bkg)

    # Model vs observed profiles of each likelihood
    likelihood = dict()
    if fit_SB:
        likelihood['sb'] = []
        for i, (this_sb_data, valid_sb) in enumerate(zip(Mhyd.sb_data, dict_utils['valid_sb_list'])):
            if fit_bkg:
                band = _posterior_band(trace, 'cts_model')
                obs = this_sb_data.counts
                err = np.sqrt(np.maximum(this_sb_data.counts, 1.))
            else:
                band = _posterior_band(trace, f'sb_model_{i}')
                obs = this_sb_data.sb
                err = this_sb_data.esb
            likelihood['sb'].append(_likelihood_dataset(this_sb_data, obs, err, band, _fit_mask(valid_sb, len(obs))))
    if fit_kT:
        likelihood['kt'] = []
        for i, (this_spec_data, valid_spec) in enumerate(zip(Mhyd.spec_data, dict_utils['valid_spec_list'])):
            band = _posterior_band(trace, f'tproj_{i}')
            mask = _fit_mask(valid_spec, len(this_spec_data.temp_x))
            likelihood['kt'].append(_likelihood_dataset(this_spec_data, this_spec_data.temp_x, this_spec_data.errt_x, band, mask))
    if fit_Z_Ab:
        likelihood['metal'] = []
        for i, (this_spec_data, index_x, valid_spec) in enumerate(zip(Mhyd.spec_data, dict_utils['index_kt_list'], dict_utils['valid_spec_list'])):
            band = _posterior_band(trace, 'zfe_model', index = index_x)
            mask = _fit_mask(valid_spec, len(this_spec_data.zfe))
            likelihood['metal'].append(_likelihood_dataset(this_spec_data, this_spec_data.zfe, this_spec_data.errt_zfe, band, mask))
    products['likelihood'] = likelihood

    # Mass profiles
    components = [('mtot_model_rref', 'Total')]
    components += [(f'm{mass_model.massmod.lower()}_model_rref', mass_model.massmod) for mass_model in models_mass]
    components += [('mgas_model_rref', 'Gas')]
    if model_mstar is not None:
        components += [('mstar_model_rref', 'Stars (model)')]
    if model_missing_mass is not None:
        components += [('m_missing_mass_model_rref', 'Missing mass')]
    products['mass'] = [dict(key = key, label = label, band = _posterior_band(trace, key)) for key, label in components]
    # mstar_m_rref: stellar mass data interpolated at rref_m, in units of Mhyd.mfact0 * 1e13
    products['mstar_data'] = dict_utils['mstar_m_rref'] * Mhyd.mfact0 * 1e13 if mstar is not None else None

    # Mass fractions, computed sample by sample.
    # Merge chains and draws; .values because each profile has its own radial dim name. Shape (npoints, nsamples)
    post = trace.posterior.stack(sample = ('chain', 'draw'))
    mtot = post['mtot_model_rref'].values
    mgas = post['mgas_model_rref'].values
    fractions = [('fgas', 'Gas fraction', mgas / mtot)]
    if model_mstar is not None:
        fractions += [('fbar', 'Baryon fraction', (mgas + post['mstar_model_rref'].values) / mtot)]
    products['fractions'] = [dict(key = key, label = label, band = np.percentile(fraction, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis = -1))
                             for key, label, fraction in fractions]

    # Overdensity quantities: {delta: {name: (lo, med, hi)}}
    products['delta'] = summarize_delta(delta_results) if delta_results is not None else dict()

    # Corner plot inputs
    models = list(models_mass) + [m for m in (model_mstar, model_missing_mass) if m is not None]
    products['corner'] = _corner_inputs(trace, models, dict_utils)

    # Thermodynamic profiles, all at the outer bin radii rout_m (entropy computed sample by sample)
    products['rout_m'] = dict_utils['rout_m']
    # Outer edge of the outermost spectral bin, in kpc
    products['rmax_spec_kpc'] = max(np.max(d.rout_x_am) for d in Mhyd.spec_data) * Mhyd.amin2kpc if Mhyd.spec_data is not None else None
    thermo = []
    if all(name in post for name in ['ne_model_rout', 't3d_out', 'press_rout']):
        ne, t3d, press = post['ne_model_rout'].values, post['t3d_out'].values, post['press_rout'].values
        for key, label, values in [('ne_model_rout', 'n$_e$ [cm$^{-3}$]', ne), ('t3d_out', 'kT [keV]', t3d),
                                   ('press_rout', 'P [keV cm$^{-3}$]', press), ('entropy_rout', 'K = kT n$_e^{-2/3}$ [keV cm$^2$]', t3d / ne ** (2. / 3.))]:
            thermo.append(dict(key = key, label = label, band = np.percentile(values, [50., 50. - 68.3 / 2., 50. + 68.3 / 2.], axis = -1)))
    products['thermo'] = thermo

    return products


def save_products(products, outfile):
    '''
    CLAUDE: Save the products of :func:`compute_products` to a FITS file, with extensions LIKELIHOOD_<KIND>_<i> (observed profile,
    model band and fit mask), MASS (mass profiles), FRACTIONS (mass fractions) and DELTA (overdensity quantities).
    The corner plot samples are not saved, they are in the trace.
    '''
    hdus = fits.HDUList([fits.PrimaryHDU()])
    head = hdus[0].header
    head['MODEL'] = products['tag']
    head['RMIN_KPC'] = products['rmin_kpc']
    head['RMAX_KPC'] = products['rmax_kpc']
    head['FIT_BKG'] = products['fit_bkg']
    if products['rmax_spec_kpc'] is not None:
        head['RMAXSPEC'] = (products['rmax_spec_kpc'], 'Outer radius of the spectral data [kpc]')

    for kind, datasets in products['likelihood'].items():
        for i, ds in enumerate(datasets):
            cols = [fits.Column(name = 'RADIUS', format = 'E', unit = 'arcmin', array = ds['x']),
                    fits.Column(name = 'RIN', format = 'E', unit = 'arcmin', array = ds['xlo']),
                    fits.Column(name = 'ROUT', format = 'E', unit = 'arcmin', array = ds['xhi']),
                    fits.Column(name = 'OBS', format = 'E', array = ds['obs']),
                    fits.Column(name = 'ERR', format = 'E', array = ds['err']),
                    fits.Column(name = 'MODEL', format = 'E', array = ds['band'][0]),
                    fits.Column(name = 'MODEL_LO', format = 'E', array = ds['band'][1]),
                    fits.Column(name = 'MODEL_HI', format = 'E', array = ds['band'][2]),
                    fits.Column(name = 'FITTED', format = 'L', array = ds['mask'])]
            hdus.append(fits.BinTableHDU.from_columns(cols, name = f'LIKELIHOOD_{kind.upper()}_{i}'))

    for extname, entries in [('MASS', products['mass']), ('FRACTIONS', products['fractions'])]:
        cols = [fits.Column(name = 'RADIUS', format = 'E', unit = 'kpc', array = products['rref_m'])]
        for entry in entries:
            for suffix, values in zip(['', '_LO', '_HI'], entry['band']):
                cols.append(fits.Column(name = entry['key'].upper() + suffix, format = 'E', array = values))
        if extname == 'MASS' and products['mstar_data'] is not None:
            cols.append(fits.Column(name = 'MSTAR_DATA', format = 'E', array = products['mstar_data']))
        hdu = fits.BinTableHDU.from_columns(cols, name = extname)
        for k, entry in enumerate(entries):
            hdu.header[f'LABEL{k}'] = (entry['label'], entry['key'].upper())
        hdus.append(hdu)

    if len(products['thermo']) > 0:
        cols = [fits.Column(name = 'RADIUS', format = 'E', unit = 'kpc', array = products['rout_m'])]
        for entry in products['thermo']:
            for suffix, values in zip(['', '_LO', '_HI'], entry['band']):
                cols.append(fits.Column(name = entry['key'].upper() + suffix, format = 'E', array = values))
        hdus.append(fits.BinTableHDU.from_columns(cols, name = 'THERMO'))

    if len(products['delta']) > 0:
        deltas = list(products['delta'].keys())
        cols = [fits.Column(name = 'DELTA', format = 'E', array = deltas)]
        for name, fmt in DELTA_COLUMNS:
            for k, suffix in enumerate(['_LO', '', '_HI']):
                cols.append(fits.Column(name = name.upper() + suffix, format = 'E', array = [products['delta'][d][name][k] for d in deltas]))
        hdus.append(fits.BinTableHDU.from_columns(cols, name = 'DELTA'))

    hdus.writeto(outfile, overwrite = True)


# ---------------------------------------------------------------------------------------------------------------------
# Plots. Each function takes a list of products: one fit is plotted in detail, several fits are overlaid, one color per fit
# ---------------------------------------------------------------------------------------------------------------------

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


def _save_figure(fig, ax, outfile, title = None):
    '''
    CLAUDE: Add the legend, apply tight_layout and save the figure to outfile, if given.
    '''
    ax.legend(title = title)
    fig.tight_layout()
    if len(fig.axes) > 1:
        # Keep the model vs observed and residual panels attached
        fig.subplots_adjust(hspace = 0)
    if outfile is not None:
        fig.savefig(outfile)
        plt.close(fig)


def _draw_data(ax, ax_res, ds, res, color, label, xshift = 1.):
    '''
    CLAUDE: Draw the observed points (fitted in color, excluded open in gray) and their residuals res. xshift multiplies the residual radii, to separate several fits.
    '''
    x, mask = ds['x'], ds['mask']
    xerr = np.vstack([x - ds['xlo'], ds['xhi'] - x])
    if ax is not None:
        ax.errorbar(x[mask], ds['obs'][mask], xerr = xerr[:, mask], yerr = ds['err'][mask], fmt = 'o', color = color, label = f'{label} fitted')
        if np.any(~mask):
            ax.errorbar(x[~mask], ds['obs'][~mask], xerr = xerr[:, ~mask], yerr = ds['err'][~mask], fmt = 'o', mfc = 'none', color = 'gray', label = f'{label} excluded')
    ax_res.errorbar(x[mask] * xshift, res[mask], yerr = 1, fmt = 'o', color = color)
    if np.any(~mask):
        ax_res.errorbar(x[~mask] * xshift, res[~mask], yerr = 1, fmt = 'o', mfc = 'none', color = 'gray' if ax is not None else color)


def _plot_likelihood_kind(products_list, kind, outfile = None):
    '''
    CLAUDE: Model vs observed plot of one likelihood kind ('sb', 'kt' or 'metal') for one or several fits.
    One fit: one color per dataset, excluded points in gray. Several fits: data in black, one color per fit for model and residuals.
    '''
    fig_title, short, ylabel, logy = LIKELIHOOD_KINDS[kind]
    single = len(products_list) == 1
    fig, ax, ax_res = _new_figure(fig_title)
    nfit = len(products_list)
    for m, prod in enumerate(products_list):
        datasets = prod['likelihood'][kind]
        for i, ds in enumerate(datasets):
            med, lo, hi = ds['band']
            res = (ds['obs'] - med) / ds['err']
            if single:
                color, label = f'C{i}', f'{short} {i}'
                ax.fill_between(ds['x'], lo, hi, color = color, alpha = 0.3, lw = 0)
                ax.plot(ds['x'], med, color = color, label = f'{label} model')
                _draw_data(ax, ax_res, ds, res, color, label)
            else:
                color = f'C{m}'
                label = prod['tag'] + (f' {short} {i}' if len(datasets) > 1 else '')
                ax.fill_between(ds['x'], lo, hi, color = color, alpha = 0.3, lw = 0)
                ax.plot(ds['x'], med, color = color, label = label)
                if m == 0:
                    # Data drawn once, in black (fitted) and gray (excluded)
                    x, mask = ds['x'], ds['mask']
                    xerr = np.vstack([x - ds['xlo'], ds['xhi'] - x])
                    data_label = f'{short} {i}' if len(datasets) > 1 else 'Data'
                    ax.errorbar(x[mask], ds['obs'][mask], xerr = xerr[:, mask], yerr = ds['err'][mask], fmt = 'o', color = 'k', label = f'{data_label} fitted')
                    if np.any(~mask):
                        ax.errorbar(x[~mask], ds['obs'][~mask], xerr = xerr[:, ~mask], yerr = ds['err'][~mask], fmt = 'o', mfc = 'none', color = 'gray',
                                    label = f'{data_label} excluded')
                # Residuals of each fit slightly shifted in radius to be distinguishable
                _draw_data(None, ax_res, ds, res, color, label, xshift = 1. + 0.02 * (m - (nfit - 1) / 2.))
    if logy:
        ax.set_yscale('log')
    ax.set_ylabel('Counts' if (kind == 'sb' and products_list[0]['fit_bkg']) else ylabel)
    _save_figure(fig, ax, outfile, title = products_list[0]['tag'] if single else None)
    return fig


def plot_likelihood(products_list, outdir, suffix = ''):
    '''
    CLAUDE: Model vs observed plots of every likelihood fitted in any of the products, saved as likelihood_<kind><suffix>.pdf in outdir.
    '''
    for kind in LIKELIHOOD_KINDS:
        with_kind = [prod for prod in products_list if kind in prod['likelihood']]
        if len(with_kind) > 0:
            _plot_likelihood_kind(with_kind, kind, outfile = os.path.join(outdir, f'likelihood_{kind}{suffix}.pdf'))


def _delta_point(summary, name):
    '''
    CLAUDE: Median and asymmetric 68.3% errors of a summarized overdensity quantity, in the format of matplotlib errorbar.
    '''
    lo, med, hi = summary[name]
    return med, [[med - lo], [hi - med]]


def _rplot_min(prod):
    '''
    CLAUDE: Inner radius of the radial plots in kpc: 0.5 * rmin, or the first model radius if rmin is 0.
    '''
    return 0.5 * prod['rmin_kpc'] if prod['rmin_kpc'] > 0 else prod['rref_m'][0]


def _mark_fit_range(ax, prod):
    '''
    CLAUDE: Set a log radial axis in kpc starting at _rplot_min and mark rmin and rmax with vertical dashed lines. To be called after drawing the profiles.
    '''
    ax.set_xscale('log')
    ax.set_xlim(left = _rplot_min(prod))
    if prod['rmin_kpc'] > 0:
        ax.axvline(prod['rmin_kpc'], color = 'gray', ls = '--')
    ax.axvline(prod['rmax_kpc'], color = 'gray', ls = '--')
    ax.set_xlabel('Radius [kpc]')


def plot_mass(products_list, outfile = None):
    '''
    CLAUDE: Mass profiles. One fit: total, each mass model, gas, stars and missing mass, stellar mass data if given, R500 and M500.
    Several fits: total mass, R500 and M500 of each fit (one color per fit), gas mass and stellar mass data of the first fit.
    The fitted radial range is marked with vertical dashed lines.
    '''
    single = len(products_list) == 1
    fig, ax = plt.subplots(figsize = (10, 8))
    shown = products_list[0]['rref_m'] >= _rplot_min(products_list[0])
    ylims = []
    for m, prod in enumerate(products_list):
        rref_m = prod['rref_m']
        kcolor = 0
        for entry in prod['mass']:
            if single:
                if entry['key'] == 'mtot_model_rref':
                    color = 'k'
                else:
                    color = f'C{kcolor}'
                    kcolor += 1
                label = entry['label']
            elif entry['key'] == 'mtot_model_rref':
                color, label = f'C{m}', prod['tag']
            elif entry['key'] == 'mgas_model_rref' and m == 0:
                color, label = 'gray', 'Gas'
            else:
                continue
            med, lo, hi = entry['band']
            ax.fill_between(rref_m, lo, hi, color = color, alpha = 0.3, lw = 0)
            ax.plot(rref_m, med, color = color, label = label)
            ylims += [lo[shown].min(), hi[shown].max()]

        if m == 0 and prod['mstar_data'] is not None:
            ax.plot(rref_m, prod['mstar_data'], 'o', mfc = 'none', color = 'gray', label = 'Stars (data)')

        if 500 in prod['delta']:
            color = 'k' if single else f'C{m}'
            r500, r500_err = _delta_point(prod['delta'][500], 'R_delta')
            m500, m500_err = _delta_point(prod['delta'][500], 'Mtot')
            ax.axvline(r500, color = color, ls = ':', label = 'R$_{500}$' if single else None)
            ax.errorbar(r500, m500, xerr = r500_err, yerr = m500_err, fmt = 'o', color = color, label = 'M$_{500}$' if single else None, capsize = 3)

    _mark_fit_range(ax, products_list[0])
    ax.set_yscale('log')
    ax.set_ylim(0.5 * min(ylims), 2 * max(ylims))
    ax.set_ylabel('M(<r) [M$_\\odot$]')
    ax.set_title('Mass profiles')
    _save_figure(fig, ax, outfile, title = products_list[0]['tag'] if single else None)
    return fig


def plot_mass_fractions(products_list, outfile = None):
    '''
    CLAUDE: Gas fraction and, when a stellar mass model was fitted, baryon fraction profiles, with the fractions within R500.
    One fit: one color per fraction. Several fits: one color per fit, gas fraction solid and baryon fraction dashed.
    The fitted radial range is marked with vertical dashed lines.
    '''
    single = len(products_list) == 1
    fig, ax = plt.subplots(figsize = (10, 8))
    shown = products_list[0]['rref_m'] >= _rplot_min(products_list[0])
    ymax = 0
    for m, prod in enumerate(products_list):
        rref_m = prod['rref_m']
        show_500 = 500 in prod['delta']
        for k, entry in enumerate(prod['fractions']):
            med, lo, hi = entry['band']
            if single:
                color, ls, label = f'C{k}', '-', entry['label']
            else:
                color, ls, label = f'C{m}', '-' if entry['key'] == 'fgas' else '--', f"{prod['tag']} {entry['label'].lower()}"
            ax.fill_between(rref_m, lo, hi, color = color, alpha = 0.3, lw = 0)
            ax.plot(rref_m, med, color = color, ls = ls, label = label)
            ymax = max(ymax, hi[shown].max())
            if show_500:
                r500, r500_err = _delta_point(prod['delta'][500], 'R_delta')
                f500, f500_err = _delta_point(prod['delta'][500], entry['key'])
                ax.errorbar(r500, f500, xerr = r500_err, yerr = f500_err, fmt = 'o', color = 'k' if single else color,
                            label = f"{entry['label']} (<R$_{{500}}$)" if single else None, capsize = 3)
        if show_500:
            ax.axvline(prod['delta'][500]['R_delta'][1], color = 'k' if single else f'C{m}', ls = ':', label = 'R$_{500}$' if single else None)

    _mark_fit_range(ax, products_list[0])
    ax.set_ylim(0, 1.1 * ymax)
    ax.set_ylabel('M / M$_{tot}$ (<r)')
    ax.set_title('Mass fractions')
    _save_figure(fig, ax, outfile, title = products_list[0]['tag'] if single else None)
    return fig


def _prior_pdf(prior_type, mu, sigma, lower, upper, x):
    '''
    CLAUDE: Density at x of a 'Uniform' (lower, upper), 'Normal' (mu, sigma) or 'TruncatedNormal' (mu, sigma, lower, upper) prior,
    as defined in :meth:`hydromass.model.Model.pm_priors`.
    '''
    if prior_type == 'TruncatedNormal':
        return stats.truncnorm.pdf(x, (lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma)
    elif prior_type == 'Normal':
        return stats.norm.pdf(x, loc = mu, scale = sigma)
    else:
        return stats.uniform.pdf(x, loc = lower, scale = upper - lower)


def plot_corner(products_list, outfile = None):
    '''
    CLAUDE: Corner plot (based on :func:`hydromass.contour.make_corner`) of the free mass model parameters and logp0.
    One fit: the prior of each parameter is drawn as a solid red line on the 1D posteriors on the diagonal; flat priors are drawn at half the height of the posterior peak.
    Several fits: the posteriors are overlaid, one color per fit, without priors; parameters missing in a fit are shown for the other fits only.
    '''
    single = len(products_list) == 1
    samples_list, all_names, tags = [], [], []
    for prod in products_list:
        corner = prod['corner']
        if len(corner['names']) == 0:
            continue
        mcsample = MCSamples(samples = corner['samples'], names = corner['names'], labels = corner['labels'], ranges = corner['ranges'],
                             label = prod['tag'])
        mcsample.updateSettings({'contours': [0.6827, 0.9545], 'smooth_scale_2D': 3, 'smooth_scale_1D': 3})
        samples_list.append(mcsample)
        tags.append(prod['tag'])
        all_names += [name for name in corner['names'] if name not in all_names]

    if len(samples_list) == 0:
        return None

    g = plots.get_subplot_plotter(width_inch = 4)
    g.settings.axes_fontsize = 30
    g.settings.axes_labelsize = 30
    g.settings.legend_fontsize = 30
    g.settings.title_limit_fontsize = 30
    g.settings.fig_width_inch = 30
    g.settings.num_plot_contours = 3
    g.settings.alpha_factor_contour_lines = 0
    g.settings.norm_1d_density = True
    g.settings.line_styles = 'tab10'
    g.settings.alpha_filled_add = 0.5

    g.triangle_plot(samples_list, all_names, filled = False, title_limit = 1 if single else 0,
                    contour_colors = [f'C{m}' for m in range(len(samples_list))], legend_labels = None if single else tags)

    if single:
        # Prior on the diagonal, normalized as the 1D posterior densities (norm_1d_density).
        # Flat priors are instead drawn at half the height of the posterior peak, to be visible
        for k, (prior_type, mu, sigma, lower, upper) in enumerate(products_list[0]['corner']['priors']):
            ax = g.subplots[k, k]
            post_max = max(line.get_ydata().max() for line in ax.get_lines())
            x = np.linspace(*ax.get_xlim(), 500)
            prior = _prior_pdf(prior_type, mu, sigma, lower, upper, x)
            if prior_type == 'Uniform':
                prior = prior / prior.max() * 0.5 * post_max
            ax.plot(x, prior, color = 'red', ls = '-', lw = 3)
            ax.set_ylim(top = max(ax.get_ylim()[1], 1.05 * prior.max()))
        g.fig.suptitle(products_list[0]['tag'], fontsize = 40)

    if outfile is not None:
        g.fig.savefig(outfile)
        plt.close(g.fig)
    return g


def plot_thermodynamics(products_list, outfile = None):
    '''
    CLAUDE: Multipage pdf with the 3D density, temperature, pressure and entropy profiles at the outer bin radii rout_m, one quantity per page.
    The outer radius of the spectral data (dotted) and R500 (dash-dotted) are marked.
    One fit: one band per page. Several fits: one color per fit. The fitted radial range is marked with vertical dashed lines.
    '''
    single = len(products_list) == 1
    with_thermo = [prod for prod in products_list if len(prod['thermo']) > 0]
    if len(with_thermo) == 0:
        return

    figs = []
    for k, entry0 in enumerate(with_thermo[0]['thermo']):
        fig, ax = plt.subplots(figsize = (10, 8))
        rplot_min = _rplot_min(with_thermo[0])
        ylims = []
        for m, prod in enumerate(with_thermo):
            rout_m = prod['rout_m']
            med, lo, hi = prod['thermo'][k]['band']
            color = 'C0' if single else f'C{m}'
            ax.fill_between(rout_m, lo, hi, color = color, alpha = 0.3, lw = 0)
            ax.plot(rout_m, med, color = color, label = prod['tag'])
            shown = rout_m >= rplot_min
            ylims += [lo[shown].min(), hi[shown].max()]
        if with_thermo[0]['rmax_spec_kpc'] is not None:
            ax.axvline(with_thermo[0]['rmax_spec_kpc'], color = 'gray', ls = ':', label = 'Max spectral radius')
        for m, prod in enumerate(with_thermo):
            if 500 in prod['delta']:
                ax.axvline(prod['delta'][500]['R_delta'][1], color = 'k' if single else f'C{m}', ls = '-.',
                           label = 'R$_{500}$' if single else f"R$_{{500}}$ {prod['tag']}")
        _mark_fit_range(ax, with_thermo[0])
        if entry0['key'] != 't3d_out':
            ax.set_yscale('log')
            ax.set_ylim(0.5 * min(ylims), 2 * max(ylims))
        else:
            ax.set_ylim(0.9 * min(ylims), 1.1 * max(ylims))
        ax.set_ylabel(entry0['label'])
        ax.set_title('Thermodynamic profiles')
        _save_figure(fig, ax, None)
        figs.append(fig)

    if outfile is not None:
        with PdfPages(outfile) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
    return figs


def plot_fits(fits_list, outdir = None):
    '''
    CLAUDE: Make all the plots (likelihood_<kind>.pdf, mass.pdf, mass_fractions.pdf, corner.pdf, thermodynamics.pdf) and the overdensity table (results.dat)
    for one or several fits, overlaying the fits in the same plots.
    Each element of fits_list is either a products dictionary from :func:`compute_products` or an :class:`hydromass.mhyd.Mhyd` object
    on which :meth:`hydromass.mhyd.Mhyd.fit` was run (its products are in Mhyd.fit_products), e.g. deep copies of the same Mhyd object
    fitted with different models.

    :param fits_list: list of products or fitted Mhyd objects
    :param outdir: output directory. Default: the directory of the first Mhyd object, or the current directory for products
    '''
    products_list = [getattr(item, 'fit_products', item) for item in fits_list]
    if outdir is None:
        outdir = getattr(fits_list[0], 'dir', '.')
    os.makedirs(outdir, exist_ok = True)

    plot_likelihood(products_list, outdir)
    plot_mass(products_list, outfile = os.path.join(outdir, 'mass.pdf'))
    plot_mass_fractions(products_list, outfile = os.path.join(outdir, 'mass_fractions.pdf'))
    plot_corner(products_list, outfile = os.path.join(outdir, 'corner.pdf'))
    plot_thermodynamics(products_list, outfile = os.path.join(outdir, 'thermodynamics.pdf'))
    with_delta = {prod['tag']: prod['delta'] for prod in products_list if len(prod['delta']) > 0}
    if len(with_delta) > 0:
        write_delta_table(with_delta, os.path.join(outdir, 'results.dat'))
