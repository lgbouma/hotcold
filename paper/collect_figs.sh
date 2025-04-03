r="/Users/luke/Dropbox/proj/cpv/results"

##########################################
# figure1: TIC 141146667 is a complex periodic variable.
# still: c1265-1268, s48, 2022-02-16 to 2022-02-17
cp $r/movie_phase_timegroups/TIC_141146667/TIC_141146667_0081_2min_phase_timegroups.pdf figures/f1.pdf

# Animated supporting: Movie of S41, S48, S75 TESS photometry, cycle-grouped
cp $r/movie_phase_timegroups/TIC_141146667/movie_TIC1411_flux_phase_wob.mov movies/m1.mov

##########################################
# figure2: Hydrogen emission from circumstellar plasma orbiting TIC 141146667.
# -still: index 0012.  t=3.1hr
cp $r/movie_sixpanel_specriver/TIC_141146667_science/specriver_141146667_Hα_0012_2min_remove25pct_normbyveq_showlinecoresum.pdf figures/f2.pdf

# Animated supporting: Movie of spectral timeseries
cp $r/movie_sixpanel_specriver/TIC_141146667_science_wob/TIC141146667_sixpanel.mov movies/m2.mov

##########################################
# supplementary

# sf1: The data are nearly but not exactly simultaneous
cp $r/tic1411_obsepoch/tic1411_obsepoch.pdf figures/sf1.pdf

r="/Users/luke/Dropbox/proj/hotcold/drivers/results"
cp $r/li_ew/li_vs_population.pdf figures/sf2.pdf

cp $r/hrd/hrd_smalllim_dereddened.pdf figures/sf3.pdf

cp $r/sed_fit/TIC_141146667/plots_irexcess/SED.pdf figures/sf4.pdf

cp $r/lineevolnpanel/TIC141146667_j537_lineevolnpanel_removecosmic.pdf figures/sf5.pdf

cp $r/halpha_to_rv_timerseries/halpha_stack_3clumps.pdf figures/sf6.pdf

cp $r/halpha_to_rv_timerseries/combined_circular_fit.pdf figures/sf7.pdf
