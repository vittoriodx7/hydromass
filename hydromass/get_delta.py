import numpy as np

__all__ = ['calc_delta_quantities', 'summarize_delta', 'write_delta_table']


def _profile_samples(post, name):
    '''
    CLAUDE: Samples of a profile from the stacked posterior, shape (nsamples, npoints).
    '''
    return post[name].values.T


def _at_crossing(profile, j, t):
    '''
    CLAUDE: Log-log interpolation of each sample profile between rref_m[j - 1] and rref_m[j], with fractional position t in log radius.
    '''
    rows = np.arange(profile.shape[0])
    lo = np.log(profile[rows, j - 1])
    hi = np.log(profile[rows, j])
    return np.exp(lo + t * (hi - lo))


def calc_delta_quantities(Mhyd, trace, dict_utils, model_mstar = None, mstar = None, deltas = (2500, 1000, 500, 200), outfile = None):
    '''
    CLAUDE: For each overdensity Delta (with respect to the critical density) and for each posterior sample, find R_delta where the mean
    total density 3 Mtot(<r) / (4 pi r^3) equals Delta * rho_c(z), and compute M_delta = 4/3 pi R_delta^3 Delta rho_c, Mgas, Mstar,
    fgas = Mgas / M_delta, fstar = Mstar / M_delta and fbar = (Mgas + Mstar) / M_delta at R_delta.
    Profiles are interpolated in log-log between the two radii of dict_utils['rref_m'] around the crossing.
    Mstar is taken from the trace if model_mstar or mstar is given, otherwise it is 0.

    :param Mhyd: :class:`hydromass.mhyd.Mhyd` object
    :param trace: InferenceData returned by the fit
    :param dict_utils: dictionary of the fit setup, containing 'rref_m' in kpc
    :param model_mstar: stellar mass model passed to the fit. Default None
    :param mstar: stellar mass profile passed to the fit. Default None
    :param deltas: list of overdensities. Default (2500, 1000, 500, 200)
    :param outfile: if given, the medians and the lower and upper errors (distance of the 15.85 and 84.15 percentiles from the median) are written to this .dat file. Default None
    :return: dictionary {delta: {'R_delta', 'Mtot', 'Mgas', 'Mstar', 'fgas', 'fstar', 'fbar'}} with one value per posterior sample (nan if R_delta is outside rref_m)
    '''
    rref_m = dict_utils['rref_m']  # kpc
    post = trace.posterior.stack(sample = ('chain', 'draw'))

    mtot = _profile_samples(post, 'mtot_model_rref')
    mgas = _profile_samples(post, 'mgas_model_rref')
    if model_mstar is not None or mstar is not None:
        mstar_prof = _profile_samples(post, 'mstar_model_rref')
    else:
        mstar_prof = np.zeros_like(mtot)

    # Mean enclosed density in units of the critical density; rho_cz is in Msun / Mpc^3
    vol = 4. / 3. * np.pi * (rref_m / 1000.) ** 3
    delta_prof = mtot / (vol * Mhyd.rho_cz)

    results = dict()
    for delta in deltas:
        # First radius (going outwards) where the mean density drops below delta
        below = delta_prof < delta
        j = np.argmax(below, axis = 1)
        valid = below.any(axis = 1) & (j > 0)
        j[~valid] = 1

        rows = np.arange(len(j))
        logd_lo = np.log(delta_prof[rows, j - 1])
        logd_hi = np.log(delta_prof[rows, j])
        t = (np.log(delta) - logd_lo) / (logd_hi - logd_lo)

        r_delta = _at_crossing(np.broadcast_to(rref_m, mtot.shape), j, t)
        m_delta = 4. / 3. * np.pi * (r_delta / 1000.) ** 3 * delta * Mhyd.rho_cz
        mgas_delta = _at_crossing(mgas, j, t)
        mstar_delta = _at_crossing(mstar_prof, j, t) if (model_mstar is not None or mstar is not None) else np.zeros_like(r_delta)

        res = {
            'R_delta': r_delta,
            'Mtot': m_delta,
            'Mgas': mgas_delta,
            'Mstar': mstar_delta,
            'fgas': mgas_delta / m_delta,
            'fstar': mstar_delta / m_delta,
            'fbar': (mgas_delta + mstar_delta) / m_delta,
        }
        for key in res:
            res[key][~valid] = np.nan

        if not valid.all():
            Mhyd.logger.info(f'WARNING: R_{delta} outside the model radial range for {np.sum(~valid)}/{len(valid)} samples, ignored')

        results[delta] = res

    if outfile is not None:
        write_delta_table({None: summarize_delta(results)}, outfile)
        Mhyd.logger.info(f'Overdensity quantities saved to {outfile}')

    return results


DELTA_COLUMNS = [('Mtot', '{:10.4E}'), ('R_delta', '{:7.0f}'), ('Mgas', '{:10.4E}'), ('Mstar', '{:10.4E}'),
                 ('fgas', '{:6.4f}'), ('fstar', '{:6.4f}'), ('fbar', '{:6.4f}')]


def summarize_delta(results):
    '''
    CLAUDE: Summarize the per-sample output of :func:`calc_delta_quantities` as {delta: {name: (lo, med, hi)}}, with lo and hi the 15.85 and 84.15 percentiles (nan ignored).
    '''
    return {delta: {name: tuple(np.nanpercentile(res[name], [50. - 68.3 / 2., 50., 50. + 68.3 / 2.])) for name, fmt in DELTA_COLUMNS}
            for delta, res in results.items()}


def write_delta_table(summaries, outfile):
    '''
    CLAUDE: Write {tag: summary} (summary from :func:`summarize_delta`) as an aligned text table with median, lower and upper errors.
    A Model column with the tag is added when tags are not None, so several fits can be written in the same file.
    '''
    with_tag = any(tag is not None for tag in summaries)
    tag_width = max([len('Model')] + [len(str(tag)) for tag in summaries]) if with_tag else 0
    # Each column is as wide as the longest of its name and its formatted values, right aligned
    widths = {name: max(len('e_' + name + '_lo'), len(fmt.format(1.))) for name, fmt in DELTA_COLUMNS}
    header = (f'{"Model":<{tag_width}} ' if with_tag else '') + f'{"Delta":>5}'
    header += ''.join(f' {name:>{widths[name]}} {"e_" + name + "_lo":>{widths[name]}} {"e_" + name + "_hi":>{widths[name]}}' for name, fmt in DELTA_COLUMNS)
    lines = [header]
    # Rows sorted by descending delta, so the fits at the same delta are next to each other (stable sort keeps the order of the fits)
    rows = sorted([(tag, delta, values) for tag, summary in summaries.items() for delta, values in summary.items()], key = lambda row: -row[1])
    for tag, delta, values in rows:
        line = (f'{str(tag):<{tag_width}} ' if with_tag else '') + f'{delta:5g}'
        for name, fmt in DELTA_COLUMNS:
            lo, med, hi = values[name]
            line += f' {fmt.format(med):>{widths[name]}} {fmt.format(med - lo):>{widths[name]}} {fmt.format(hi - med):>{widths[name]}}'
        lines.append(line)
    with open(outfile, 'w') as f:
        f.write('\n'.join(lines) + '\n')
