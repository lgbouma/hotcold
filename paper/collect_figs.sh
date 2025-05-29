r="/Users/luke/Dropbox/proj/cpv/results"

##########################################
# figure1: TIC 141146667 is a complex periodic variable.
# still: c1265-1268, s48, 2022-02-16 to 2022-02-17
cp $r/movie_phase_timegroups/TIC_141146667/TIC_141146667_0081_2min_phase_timegroups.pdf f1.pdf

# Animated supporting: Movie of S41, S48, S75 TESS photometry, cycle-grouped
cp $r/movie_phase_timegroups/TIC_141146667/movie_TIC1411_flux_phase_wob.mov m1.mov

##########################################
# f2: The data are nearly but not exactly simultaneous
cp $r/tic1411_obsepoch/tic1411_obsepoch.pdf f2.pdf

##########################################
# figure3: Hydrogen emission from circumstellar plasma orbiting TIC 141146667.
# -still: index 0012.  t=3.1hr
cp $r/movie_sixpanel_specriver/TIC_141146667_science/specriver_141146667_Hα_0012_2min_remove25pct_normbyveq_showlinecoresum.pdf f3.pdf

# Animated supporting: Movie of spectral timeseries
cp $r/movie_sixpanel_specriver/TIC_141146667_science_wob/TIC141146667_sixpanel.mov m3.mov
cp $r/movie_sixpanel_specriver/TIC_141146667_science_wob/TIC141146667_sixpanel_sinusoids.mov m3_bonus.mov

##########################################
# supplementary

r="/Users/luke/Dropbox/proj/hotcold/drivers/results"

cp $r/sed_fit/TIC_141146667/plots_irexcess/SED.pdf f4.pdf

cp $r/hrd/hrd_bprp_smalllim_dereddened.pdf f5a.pdf

cp $r/li_ew/li_vs_population.pdf f5b.pdf

cp $r/halpha_to_rv_timerseries/halpha_stack_3clumps.pdf f6.pdf

cp $r/halpha_to_rv_timerseries/merged_circular_fit.pdf f7.pdf

cp $r/lineevolnpanel/TIC141146667_j537_lineevolnpanel_removecosmic.pdf f8.pdf
