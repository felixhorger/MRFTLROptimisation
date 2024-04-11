using Revise
import PyPlot as plt
import PyPlotTools
import MRIPhantoms
import MRIRecon
using JLD2
using FFTW
using Statistics
PyPlotTools.set_image()

random = 1e5 .* load("../../../20230530_Scanning/random/mrf_phasecorrected.jld2", "recon")[:, :, 100, :]
optimised = 1e5 .* load("../../../20230530_Scanning/optimised/mrf_phasecorrected.jld2", "recon")[:, :, 100, :]
mask = circshift(MRIPhantoms.ellipsoidal_shutter(size(optimised)[1:2], (90, 90)), (0, 3))
mask_maps = circshift(MRIPhantoms.ellipsoidal_shutter(size(optimised)[1:2], (86, 86)), (0, 3))
#@views mask = abs.(optimised[:, :, 1]) .> 0.5 

caspr = 1.2e5 .* load("../../../20230713_Scanning/caspr/mrf.jld2", "recon")[:, :, 108, :]
caspr_corrected = 1.2e5 .* load("../../../20230713_Scanning/caspr/phasecorrected/mrf_10.8deg.jld2", "recon")[:, :, 108, :]
@views mask_caspr = circshift(MRIPhantoms.ellipsoidal_shutter(size(caspr)[1:2], (88, 88)), (5, 2)) #abs.(caspr[:, :, 1]) .> 0.45
@views mask_caspr_maps = circshift(MRIPhantoms.ellipsoidal_shutter(size(caspr)[1:2], (86, 86)), (5, 2)) #abs.(caspr[:, :, 1]) .> 0.45
#radial = 4e7 .* load("../../../20230713_Scanning/radial/undersampled_mrf.jld2", "recon")[100, :, :, :]
radial = 1.5e5 .* load("../../../20230713_Scanning/radial/undersampled_mrf.jld2", "recon")[108, :, :, :]
#radial_spoke_undersampled = 3e7 .* load("../../../20230713_Scanning/radial/mrf_spoke_undersampled.jld2", "recon")[108, :, :, :]
caspr_simulation = 1.25 * load("../../../20230713_Scanning/simulation/recon.jld2", "x")
#fullysampled = 1.0 * load("../../../20230713_Scanning/simulation/recon.jld2", "fullysampled_recon")
@views mask_radial = abs.(radial[:, :, 1]) .> 0.5
mask_sim = circshift(MRIPhantoms.ellipsoidal_shutter(size(caspr)[1:2], (88, 88)), (5, 2))
mask_sim_maps = circshift(MRIPhantoms.ellipsoidal_shutter(size(caspr)[1:2], (86, 86)), (5, 2))

#fig, axs = plt.subplots(4, 5, figsize=(PyPlotTools.latex_column * 1.15, 5.6))
fig, axs = plt.subplots(4, 4, figsize=(0.7 * PyPlotTools.latex_column, 0.77 * 5.6))
plt.subplots_adjust(wspace=0.02, hspace=0.00)
axs[1, 1].set_title("\\hspace*{-1.5cm} Random and Optimised Uniform", ha="left", fontsize=11)
#axs[1, 2].set_title("Optimised Uniform")
axs[1, 3].set_title("CASPR", fontsize=11)
#axs[1, 4].set_title("CASPR Corrected")
axs[1, 4].set_title("Target", fontsize=11)
global image
scales = (1, 5, 30, 50)
vmin = 0
vmax = 2.5
for s = 1:4
	foreach(i -> axs[s, i].tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), 1:4)
	if s > 1
		label = "\$\\sigma_$s \\times $(scales[s])\$"
	else
		label = "\$\\sigma_$s\$"
	end
	axs[s, 1].set_ylabel(label; x=1.0)
	plot = axs[s, 1].imshow(scales[s] .* mask .* abs.(random[:, :, s]); cmap="gray", vmin, vmax)
	if s == 1
		image = plot
	end
	axs[s, 2].imshow(scales[s] .* mask .* abs.(optimised[:, :, s]); cmap="gray", vmin, vmax)
	#axs[s, 3].imshow(scales[s] .* mask_caspr .* abs.(caspr[:, :, s]); cmap="gray", vmin, vmax)
	axs[s, 3].imshow(scales[s] .* mask_caspr .* abs.(caspr_corrected[:, :, s]); cmap="gray", vmin, vmax)
	#axs[s, 3].imshow(scales[s] .* mask_radial .* abs.(radial_spoke_undersampled[:, :, s]); cmap="gray", vmin, vmax)
	#axs[s, 4].imshow(scales[s] .* mask_radial .* abs.(radial[:, :, s]); cmap="gray", vmin, vmax)#, aspect=size(radial, 2)/size(radial, 1))
	axs[s, 4].imshow(scales[s] .* mask_sim .* abs.(caspr_simulation[:, :, s]); cmap="gray", vmin, vmax)#, aspect=size(radial, 2)/size(radial, 1))
	#for i = 1:5
	#	axs[s, i].spines["top"].set_visible(false)
	#	axs[s, i].spines["bottom"].set_visible(false)
	#	axs[s, i].spines["right"].set_visible(false)
	#	axs[s, i].spines["left"].set_visible(false)
	#end
end
#ax = fig.add_axes([0.3, -0.05, 0.4, 0.025])
#cbar = plt.colorbar(image; orientation="horizontal", cax=ax)
#cbar.set_label("Signal amplitude")
plt.savefig("multichannel.eps"; dpi=200)
plt.close(fig)

#fig, axs = plt.subplots(4, 4, figsize=(PyPlotTools.latex_column, 6))
#plt.subplots_adjust(wspace=0.05, hspace=0.0)
#axs[1, 1].set_title("Random Uniform")
#axs[1, 2].set_title("Optimised Uniform")
#axs[1, 3].set_title("CASPR")
#axs[1, 4].set_title("Stack of Stars")
#global image
#scales = (1, 5, 30, 50)
#vmin = 0
#vmax = 2.5
#for s = 1:4
#	foreach(i -> axs[s, i].tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), 1:4)
#	if s > 1
#		label = "\$\\sigma_$s \\times $(scales[s])\$"
#	else
#		label = "\$\\sigma_$s\$"
#	end
#	axs[s, 1].set_ylabel(label; x=1.0)
#	plot = axs[s, 1].imshow(scales[s] .* mask .* abs.(random[:, :, s]); cmap="gray", vmin, vmax)
#	if s == 1
#		image = plot
#	end
#	axs[s, 2].imshow(scales[s] .* mask .* abs.(optimised[:, :, s]); cmap="gray", vmin, vmax)
#	axs[s, 3].imshow(scales[s] .* mask_caspr .* abs.(caspr[:, :, s]); cmap="gray", vmin, vmax)
#	#axs[s, 3].imshow(scales[s] .* mask_radial .* abs.(radial_spoke_undersampled[:, :, s]); cmap="gray", vmin, vmax)
#	axs[s, 4].imshow(scales[s] .* mask_radial .* abs.(radial[:, :, s]); cmap="gray", vmin, vmax)#, aspect=size(radial, 2)/size(radial, 1))
#	for i = 1:4
#		axs[s, i].spines["top"].set_visible(false)
#		axs[s, i].spines["bottom"].set_visible(false)
#		axs[s, i].spines["right"].set_visible(false)
#		axs[s, i].spines["left"].set_visible(false)
#	end
#end
#ax = fig.add_axes([0.3, -0.05, 0.4, 0.025])
#cbar = plt.colorbar(image; orientation="horizontal", cax=ax)
#cbar.set_label("Singular components")
#plt.savefig("multichannel.eps"; dpi=200)
#plt.close(fig)


random_maps = load("../../../20230530_Scanning/random/maps_phasecorrected.jld2", "t1", "t2")
optimised_maps = load("../../../20230530_Scanning/optimised/maps_phasecorrected.jld2", "t1", "t2")
caspr_maps = load("../../../20230713_Scanning/caspr/maps.jld2", "t1", "t2")
caspr_corrected_maps = load("../../../20230713_Scanning/caspr/phasecorrected/maps10.8deg.jld2", "t1", "t2")
simulation_maps = load("../../../20230713_Scanning/simulation/maps.jld2", "t1", "t2")
radial_maps = load("../../../20230713_Scanning/radial/undersampled_maps.jld2", "t1", "t2")

# Statistics
PyPlotTools.rcParams["text.latex.preamble"] = """
	\\usepackage{amsmath}
	\\usepackage{amssymb}
	\\usepackage{bm}
	\\usepackage{siunitx}
	\\DeclareSIUnit\\pixel{px}
	\\newcommand{\\SIm}[2]{\\SI[round-mode=figures,round-precision=3]{#1}{#2}}
	\\newcommand{\\SIv}[2]{\\SI[round-mode=figures,round-precision=2]{#1}{#2}}
"""

random_t12 = (
	"\$T_1 = \\SIm{542.797543834}{}\\pm\\SIv{31.6741388729725}{\\milli\\second}\$",
	"\$T_2 = \\SIm{34.02373736846319}{}\\pm\\SIv{3.407330746925396}{\\milli\\second}\$"
)
optimised_t12 = (
	"\$T_1 = \\SIm{522.7405535423999}{}\\pm\\SIv{25.510296690799738}{\\milli\\second}\$",
	"\$T_2 = \\SIm{33.59134076170386}{}\\pm\\SIv{2.926435046862577}{\\milli\\second}\$"
)
caspr_t12 = (
	"\$T_1 = \\SIm{523.2203761526332}{}\\pm\\SIv{13.038234638531145}{\\milli\\second}\$",
	"\$T_2 = \\SIm{33.51278939838553}{}\\pm\\SIv{1.6264345523982513}{\\milli\\second}\$"
)
caspr_corrected_t12 = (
	"\$T_1 = \\SIm{523.760099137766}{}\\pm\\SIv{12.502606066527116}{\\milli\\second}\$",
	"\$T_2 = \\SIm{33.54916683201231}{}\\pm\\SIv{1.4389737419621162}{\\milli\\second}\$"
)
radial_t12 = (
	"\$T_1 = \\SIm{536.1606045732864}{}\\pm\\SIv{15.481550112672178}{\\milli\\second}\$",
	"\$T_2 = \\SIm{32.37910170323416}{}\\pm\\SIv{2.1020239565404966}{\\milli\\second}\$"
)
simulation_t12 = (
	"\$T_1 = \\SIm{517.1567033109998}{}\\pm\\SIv{5.574113886839618}{\\milli\\second}\$",
	"\$T_2 = \\SIm{34.35009432121252}{}\\pm\\SIv{0.7607240225349557}{\\milli\\second}\$"
)

function add_stats(ax, stats)
	ax.text(5, 5, stats; color="white", ha="left", fontsize=8)
	#transform=ax.transAxes, 
	return
end
fig, axs = plt.subplots(2, 4, figsize=(0.8 * PyPlotTools.latex_column, 2.3))
plt.subplots_adjust(wspace=-0.2, hspace=0.04)
global image
vmins = (100, 20, 0.5)
vmaxs = (1200, 80, 1.75)
axs[1, 1].set_title("\\hspace*{-1.3cm} Random and Optimised Uniform"; ha="left", fontsize=11)
#axs[1, 2].set_title("Optimised Uniform"; fontsize=11)
axs[1, 3].set_title("CASPR"; fontsize=11)
axs[1, 4].set_title("Target"; fontsize=11)
#axs[1, 4].set_title("CASPR Corrected")
#axs[1, 4].set_title("Stack of Stars")
cmaps = ("inferno", "viridis")
for s = 1:2
	foreach(i -> axs[s, i].tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), 1:4)
	plot = axs[s, 1].imshow(circshift(mask_maps .* random_maps[s][100, :, :], (6, 0)); cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 1], plot; size="8%", phantom=true)
	add_stats(axs[s, 1], random_t12[s])
	plot = axs[s, 2].imshow(circshift(mask_maps .* optimised_maps[s][100, :, :], (6, 0)); cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 2], plot; size="8%", phantom=true)
	add_stats(axs[s, 2], optimised_t12[s])
	#plot = axs[s, 3].imshow(mask_caspr .* caspr_maps[s][108, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	#cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 3], plot; size="8%", phantom=true)
	#add_stats(axs[s, 3], caspr_t12[s])
	plot = axs[s, 3].imshow(mask_caspr_maps .* caspr_corrected_maps[s][108, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 3], plot; size="8%", phantom=true)
	add_stats(axs[s, 3], caspr_corrected_t12[s])
	#plot = axs[s, 4].imshow(mask_radial .* radial_maps[s][108, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	plot = axs[s, 4].imshow(mask_sim_maps .* simulation_maps[s]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 4], plot; size="8%")
	add_stats(axs[s, 4], simulation_t12[s])
	if s == 1
		cbar.set_label("\$T_1\$ [ms]"; rotation=-90, labelpad=7)
	elseif s == 2
		cbar.set_label("\$T_2\$ [ms]"; rotation=-90, labelpad=15)
	#else
	#	cbar.set_label("relative \$B_1^+\$ [1]"; rotation=-90, labelpad=15)
	end
	for i = 1:4
		axs[s, i].spines["top"].set_visible(false)
		axs[s, i].spines["bottom"].set_visible(false)
		axs[s, i].spines["right"].set_visible(false)
		axs[s, i].spines["left"].set_visible(false)
	end
end
plt.savefig("maps.eps"; dpi=200)
plt.close(fig)


# Compute statistics in ROI
roi = MRIPhantoms.ellipsoidal_shutter(size(random_maps[1]), (70, 70, 70))
random_stats = Vector{NTuple{2, Float64}}(undef, 2)
optimised_stats = Vector{NTuple{2, Float64}}(undef, 2)
caspr_stats = Vector{NTuple{2, Float64}}(undef, 2)
radial_stats = Vector{NTuple{2, Float64}}(undef, 2)
for (maps, stats) in ((random_maps, random_stats), (optimised_maps, optimised_stats), (caspr_maps, caspr_stats), (radial_maps, radial_stats))
	for m = 1:2
		map_roi = maps[m][roi]
		roi_mean = mean(map_roi)
		roi_std = stdm(map_roi, roi_mean)
		stats[m] = (roi_mean, roi_std)
	end
end

roi = MRIPhantoms.ellipsoidal_shutter(size(simulation_maps[1]), (70, 70))
simulation_stats = Vector{NTuple{2, Float64}}(undef, 2)
for m = 1:2
	map_roi = simulation_maps[m][roi]
	roi_mean = mean(map_roi)
	roi_std = stdm(map_roi, roi_mean)
	simulation_stats[m] = (roi_mean, roi_std)
end

println("$random_stats, $optimised_stats, $caspr_stats, $radial_stats, $simulation_stats")




function add_stats(ax, stats)
	ax.text(5, 5, stats; color="white", ha="left", fontsize=8)
	#transform=ax.transAxes, 
	return
end
fig, axs = plt.subplots(2, 4, figsize=(0.6 * PyPlotTools.latex_column, 3.5))
plt.subplots_adjust(wspace=0.0, hspace=-0.2)
global image
vmins = (100, 20, 0.5)
vmaxs = (1200, 80, 1.75)
axs[1, 1].set_title("Random and Optimised Uniform"; ha="left", fontsize=11)
#axs[1, 2].set_title("Optimised Uniform"; fontsize=11)
axs[1, 3].set_title("CASPR"; fontsize=11)
axs[1, 4].set_title("Stack of Stars"; fontsize=11)
cmaps = ("inferno", "viridis")
for s = 1:2
	foreach(i -> axs[s, i].tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), 1:4)
	plot = axs[s, 1].imshow(mask .* random_maps[s][100, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 1], plot; size="8%", phantom=true)
	add_stats(axs[s, 1], random_t12[s])
	plot = axs[s, 2].imshow(mask .* optimised_maps[s][100, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 2], plot; size="8%", phantom=true)
	add_stats(axs[s, 2], optimised_t12[s])
	plot = axs[s, 3].imshow(mask_caspr .* caspr_maps[s][108, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 3], plot; size="8%", phantom=true)
	add_stats(axs[s, 3], caspr_t12[s])
	plot = axs[s, 4].imshow(mask_radial .* radial_maps[s][108, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 4], plot; size="8%")
	add_stats(axs[s, 4], simulation_t12[s])
	if s == 1
		cbar.set_label("\$T_1\$ [ms]"; rotation=-90, labelpad=15)
	elseif s == 2
		cbar.set_label("\$T_2\$ [ms]"; rotation=-90, labelpad=15)
	#else
	#	cbar.set_label("relative \$B_1^+\$ [1]"; rotation=-90, labelpad=15)
	end
	for i = 1:4
		axs[s, i].spines["top"].set_visible(false)
		axs[s, i].spines["bottom"].set_visible(false)
		axs[s, i].spines["right"].set_visible(false)
		axs[s, i].spines["left"].set_visible(false)
	end
end
plt.savefig("maps_radial.eps"; dpi=200)
plt.close(fig)

