using Revise
import PyPlot as plt
import PyPlotTools
import MRIPhantoms
import MRIRecon
using JLD2
using FFTW
using Statistics
import Noise
PyPlotTools.set_image()

random = 8e4 .* load("../../../20240219_Scanning/random/mrf_single_channel.jld2", "recon")[:, :, 100, :, :]
optimised = 8e4 .* load("../../../20240219_Scanning/optimised/mrf_single_channel.jld2", "recon")[:, :, 100, :, :]
caspr = 8e4 .* load("../../../20240219_Scanning/caspr/mrf_single_channel.jld2", "recon")[:, :, 100, :, :]
caspr = circshift(caspr, (-8, 3, 0))
#sim = 0.6 .* load("../../../20240219_Scanning/simulation/sim.jld2", "recon") #zeros(size(caspr, 1), size(caspr, 2), size(caspr, 4))
sim = 3e0 .* load("../../../20240219_Scanning/simulation/fullysampled.jld2", "fullysampled")
GC.gc()

# Plot single channel components
channels = (2,)
fig, axs = plt.subplots(4, 3, figsize=(0.525 * PyPlotTools.latex_column, 0.77 * 5.6))
plt.subplots_adjust(wspace=0.03, hspace=0.0)
fig.suptitle("Phantom Experiment"; y=1.12, fontsize=16)
axs[1, 1].set_title("Rand. Uniform"; fontsize=11)
axs[1, 2].set_title("Opt. Uniform"; fontsize=11)
axs[1, 3].set_title("CASPR"; fontsize=11)
#axs[1, 4].set_title("Target"; fontsize=11)
global image
scales = (1, 8, 16, 20)
vmin = 0
vmax = 1.5
for s = 1:4
	foreach(i -> axs[s, i].tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), 1:3)
	if s > 1
		label = "\$\\sigma_$s \\times $(scales[s])\$"
	else
		label = "\$\\sigma_$s\$"
	end
	#axs[s, 1].set_ylabel(label; x=1.0)
	plot = axs[s, 1].imshow(scales[s] .* abs.(random[:, :, channels[1], s]); cmap="gray", vmin, vmax)
	if s == 1
		image = plot
	end
	axs[s, 2].imshow(scales[s] .* abs.(optimised[:, :, channels[1], s]); cmap="gray", vmin, vmax)
	axs[s, 3].imshow(scales[s] .* abs.(caspr[:, :, channels[1], s]); cmap="gray", vmin, vmax)
	#axs[s, 4].imshow(scales[s] .* abs.(sim[:, :, s]); cmap="gray", vmin, vmax)
	for i = 1:3
		axs[s, i].spines["top"].set_visible(false)
		axs[s, i].spines["bottom"].set_visible(false)
		axs[s, i].spines["right"].set_visible(false)
		axs[s, i].spines["left"].set_visible(false)
	end
end
plt.savefig("single_channel.eps"; dpi=200)
plt.close(fig)

# STOP


# Single channel row
# Plot two images with one array
split_factor = 0.05
images = Matrix{Float64}(undef, size(optimised, 1), floor(Int, (2 + split_factor) * size(optimised, 2)))
j = 1 + floor(Int, (1+split_factor) * size(optimised, 2))
images[:, size(optimised, 2)+1:j-1] .= 1.5
length(j:size(images, 2))

# Plot single channel components
channels = (1, 3)
fig, axs = plt.subplots(2, 3, figsize=(PyPlotTools.latex_column, 3))
plt.subplots_adjust(wspace=0.05, hspace=-0.5)
axs[1, 1].set_ylabel("Random")
axs[2, 1].set_ylabel("Optimised")
global image
scales = (1, 5, 15)
vmin = 0
vmax = 1.5
for s = 1:3
	foreach(i -> axs[i, s].tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), 1:2)
	if s > 1
		label = "\$\\sigma_$s \\times $(scales[s])\$"
	else
		label = "\$\\sigma_$s\$"
	end
	axs[1, s].set_title(label; y=1.08)
	images[:, 1:size(random, 2)] .= scales[s] .* abs.(random[:, :, channels[1], s])
	images[:, j:end] .= scales[s] .* abs.(random[:, :, channels[2], s])
	plot = axs[1, s].imshow(images; cmap="gray", vmin, vmax)
	if s == 1
		image = plot
	end
	images[:, 1:size(optimised, 2)] .= scales[s] .* abs.(optimised[:, :, channels[1], s])
	images[:, j:end] .= scales[s] .* abs.(optimised[:, :, channels[2], s])
	axs[2, s].imshow(images; cmap="gray", vmin, vmax)
	for i = 1:2
		axs[i, s].spines["top"].set_visible(false)
		axs[i, s].spines["bottom"].set_visible(false)
		axs[i, s].spines["right"].set_visible(false)
		axs[i, s].spines["left"].set_visible(false)
	end
end
ax = fig.add_axes([0.3, 0.1, 0.4, 0.04])
cbar = plt.colorbar(image; orientation="horizontal", cax=ax)
cbar.set_label("Singular components")
plt.savefig("single_channel.eps"; dpi=200)
plt.close(fig)


# Multi-channel components
multi_random = 1e5 .* load("../random/mrf_phasecorrected.jld2", "recon")[:, :, 100, :]
multi_optimised = 1e5 .* load("../optimised/mrf_phasecorrected.jld2", "recon")[:, :, 100, :]

fig, axs = plt.subplots(2, 3, figsize=(0.5PyPlotTools.latex_column, 3))
plt.subplots_adjust(wspace=0.05, hspace=-0.5)
axs[1, 1].set_ylabel("Random Uniform")
axs[2, 1].set_ylabel("Optimised Uniform")
global image
scales = (1, 5, 15)
vmin = 0
vmax = 2.5
for s = 1:3
	foreach(i -> axs[i, s].tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), 1:2)
	if s > 1
		label = "\$\\sigma_$s \\times $(scales[s])\$"
	else
		label = "\$\\sigma_$s\$"
	end
	axs[1, s].set_title(label; y=1.08)
	plot = axs[1, s].imshow(scales[s] .* abs.(multi_random[:, :, s]); cmap="gray", vmin, vmax)
	if s == 1
		image = plot
	end
	axs[2, s].imshow(scales[s] .* abs.(multi_optimised[:, :, s]); cmap="gray", vmin, vmax)
	for i = 1:2
		axs[i, s].spines["top"].set_visible(false)
		axs[i, s].spines["bottom"].set_visible(false)
		axs[i, s].spines["right"].set_visible(false)
		axs[i, s].spines["left"].set_visible(false)
	end
end
ax = fig.add_axes([0.2, 0.05, 0.6, 0.04])
cbar = plt.colorbar(image; orientation="horizontal", cax=ax)
cbar.set_label("Singular components")
plt.savefig("multi_channel.eps"; dpi=200)
plt.close(fig)


# Maps
random_maps = load("../random/maps_phasecorrected.jld2", "t1", "t2", "b1")
optimised_maps = load("../optimised/maps_phasecorrected.jld2", "t1", "t2", "b1")
mask = abs.(multi_optimised[:, :, 1]) .> 0.5
#imshow(mask)

fig, axs = plt.subplots(2, 2, figsize=(0.5PyPlotTools.latex_column, 3))
plt.subplots_adjust(wspace=0.05, hspace=0.15)
global image
scales = (1, 5, 15)
vmin = 0
vmax = 2.5
vmins = (100, 20, 0.5)
vmaxs = (1200, 80, 1.75)
axs[1, 1].set_title("Random"; y=1.05)
axs[1, 2].set_title("Optimised"; y=1.05)
cmaps = ("inferno", "viridis")
for s = 1:2
	foreach(i -> axs[s, i].tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), 1:2)
	plot = axs[s, 1].imshow(mask .* random_maps[s][100, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 1], plot; phantom=true)
	plot = axs[s, 2].imshow(mask .* optimised_maps[s][100, :, :]; cmap=cmaps[s], vmin=vmins[s], vmax=vmaxs[s])
	cax, cbar = PyPlotTools.add_colourbar(fig, axs[s, 2], plot)
	if s == 1
		cbar.set_label("\$T_1\$ [ms]"; rotation=-90, labelpad=15)
	elseif s == 2
		cbar.set_label("\$T_2\$ [ms]"; rotation=-90, labelpad=15)
	#else
	#	cbar.set_label("relative \$B_1^+\$ [1]"; rotation=-90, labelpad=15)
	end
	for i = 1:2
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
for (maps, stats) in ((random_maps, random_stats), (optimised_maps, optimised_stats))
	for m = 1:2
		map_roi = maps[m][roi]
		roi_mean = mean(map_roi)
		roi_std = stdm(map_roi, roi_mean)
		stats[m] = (roi_mean, roi_std)
	end
end
println("$random_stats, $optimised_stats")


# Reconstructed k-space
random_parallel = load("../random/mrf.jld2", "recon")
optimised_parallel = load("../optimised/mrf.jld2", "recon")
@views random_kspace = fftshift(fft(random_parallel[:, :, 100, :], 1:2), 1:2)
@views optimised_kspace = fftshift(fft(optimised_parallel[:, :, 100, :], 1:2), 1:2)

fig, axs = plt.subplots(1, 4; figsize=(PyPlotTools.latex_column, 3))
plt.subplots_adjust(wspace=0.05)
foreach(ax -> ax.tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), axs)
scale = (1, 5, 15, 20)
for s = 1:4
	#global plot = axs[s].imshow(log10.(1e-10 .+ scale[s] .* abs.(optimised_kspace[:, :, s])); vmin=-3, vmax=-1)
	global plot = axs[s].imshow(scale[s] .* abs.(optimised_kspace[60:210-60, 60:224-60, s]); cmap="gray", vmin=0, vmax=3e-2)
	axs[s].set_title("\$\\sigma_$s \\times $(scale[s])\$")
end
ax = fig.add_axes([0.3, 0.15, 0.4, 0.04])
cbar = plt.colorbar(plot; orientation="horizontal", cax=ax)
cbar.set_label("\$k\$-space")
#cbar.set_ticks(-3:0.5:-1)
plt.savefig("kspace.eps"; dpi=200)
plt.close(fig)




@views random_single_kspace = fftshift(fft(random, 1:2), 1:2)
@views optimised_single_kspace = fftshift(fft(optimised, 1:2), 1:2)

fig, axs = plt.subplots(2, 4; figsize=(PyPlotTools.latex_column, 4))
plt.subplots_adjust(wspace=0.05, hspace=-0.4)
foreach(ax -> ax.tick_params(labelbottom=false, bottom=false, labelleft=false, left=false), axs)
scale = (1, 5, 15, 20)
for s = 1:4
	#global plot = axs[s].imshow(log10.(1e-10 .+ scale[s] .* abs.(optimised_kspace[:, :, s])); vmin=-3, vmax=-1)
	global plot = axs[1, s].imshow(scale[s] .* abs.(random_single_kspace[60:210-60, 60:224-60, 1, s]); cmap="gray", vmin=0, vmax=2.0e+3)
	axs[1, s].set_title("\$\\sigma_$s \\times $(scale[s])\$")
	global plot = axs[2, s].imshow(scale[s] .* abs.(optimised_single_kspace[60:210-60, 60:224-60, 1, s]); cmap="gray", vmin=0, vmax=2.0e+3)
end
axs[1, 1].set_ylabel("Initial")
axs[2, 1].set_ylabel("Optimised")
ax = fig.add_axes([0.3, 0.05, 0.4, 0.04])
cbar = plt.colorbar(plot; orientation="horizontal", cax=ax)
cbar.set_label("\$k\$-space")
#cbar.set_ticks(0:1)
plt.savefig("single_channel_kspace.eps"; dpi=200)
plt.close(fig)

