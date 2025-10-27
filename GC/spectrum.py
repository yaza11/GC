from chem_spectra.base import SpectrumBaseClass, SpectrumType

from GC.identify_standards import by_pattern_matching, predict_labels_from_fit, assign_labels_from_fit


class SpectrumGc(SpectrumBaseClass):
    spectrum_type: SpectrumType = 'GC'

    def add_std_annotations(self, spec_std: 'SpectrumGcStandard', **kwargs) -> None:
        """modify peaks annotation attribute"""
        kwargs_assign = {}
        if 'max_tol_assignment' in kwargs:
            kwargs_assign['max_tol_assignment'] = kwargs.pop('max_tol_assignment')

        # turn spec_std into pattern and pass it to by_pattern_matching
        lbls = [p.annotation for p in spec_std.peak_list.peaks]
        assert any([l is not None for l in lbls]), 'assign labels to standard first'
        wghts = [p.y for p in spec_std.peak_list.peaks]
        w_norm = max(wghts)
        wghts = [w / w_norm for w in wghts]
        pattern: dict[float, float] = {}
        for lbl, wght in zip(lbls, wghts):
            if (lbl is None) or (not lbl.startswith('C')):
                continue
            ch_length = int(lbl[1:])
            pattern[ch_length] = wght

        peak_positions = [p.x for p in self.peak_list.peaks]

        params, trafo, *axs = by_pattern_matching(
            peak_positions=peak_positions,
            peak_heights=[p.y for p in self.peak_list.peaks],
            pattern_std=pattern,
            **kwargs
        )

        predicted_labels = predict_labels_from_fit(peak_positions, trafo)
        assigned_labels, *_ = assign_labels_from_fit(predicted_labels, **kwargs_assign)
        for idx, lbl in assigned_labels.items():
            self.peak_list.peaks[idx].annotation = lbl


class SpectrumGcStandard(SpectrumGc):
    is_standard_run = True

    def annotate_peaks(self, **kwargs) -> None:
        kwargs_assign = {}
        if 'max_tol_assignment' in kwargs:
            kwargs_assign['max_tol_assignment'] = kwargs.pop('max_tol_assignment')

        peak_positions = [p.x for p in self.peak_list.peaks]
        params, trafo, *axs = by_pattern_matching(
            peak_positions=peak_positions,
            peak_heights=[p.y for p in self.peak_list.peaks],
            **kwargs
        )
        predicted_labels = predict_labels_from_fit(peak_positions, trafo)
        assigned_labels, *_ = assign_labels_from_fit(predicted_labels, **kwargs_assign)
        for idx, lbl in assigned_labels.items():
            self.peak_list.peaks[idx].annotation = lbl

    def add_std_annotations(self, spec_std: 'SpectrumGcStandard', **kwargs) -> None:
        raise NotImplementedError(
            'Cannot add annotations from other standard. If you want to assign annotations, '
            'call "annotate_peaks"'
        )

