using Revise
using JLD2
using Statistics
using FFTW
import MRIRecon
import PyPlot as plt
import PyPlotTools
PyPlotTools.set_image()
function compute_percentiles(errors, normalisation)
	return [
		quantile(dropdims(sqrt.(sum(errors.^2; dims=1) ./ sum(normalisation)); dims=1), 0.2),
		quantile(dropdims(sqrt.(sum(errors.^2; dims=1) ./ sum(normalisation)); dims=1), 0.8)
	]
end
function compute_percentiles(conditioning)
	return [quantile(conditioning, 0.2), quantile(conditioning, 0.8)]
end

num_σ = 4
names = ("Jiang2014", "Zhao2019", "Koolstra2018", "FastCycle")
fullysampled_recons = Tuple(load("$(name)_recons.jld2", "fullysampled_recon") for name in names)
lr_inversion_recons = Tuple(load("$(name)_recons.jld2", "lr_inversion_recons") for name in names)
conditionings = [
	load("optimised_sampling.jld2", "best_conditionings_$(name)") .^2
	for name in ("Jiang", "Zhao", "Koolstra", "FastCycle")
]
names_short = ("Jiang", "Zhao", "Koolstra", "Fast cycle")
schedules = ("Jiang2014_1032_recons_iter.jld2", "Zhao2019_400_recons_iter.jld2", "Koolstra2018_240_recons_iter.jld2", "FastCycle_840_recons_iter.jld2")
num_points = 7
kspaces_iter = load("../fully_sampled_criterion/" * schedules[1], "kspaces_to_sample")[1:num_points]
println([c[end] for c in conditionings])

# CASPR
caspr_errors = Vector{Float64}(undef, 4)
caspr_conditionings = Vector{NTuple{2, Float64}}(undef, 4)
for (i, name) in enumerate(("Jiang2014", "Zhao2019", "Koolstra2018", "FastCycle"))
	x, f, c, cc = load("../../../20230713_Scanning/simulation/single_channel/$(name)_recons.jld2", "lr_inversion_recons", "fullysampled_recon", "conditioning_filtered", "conditioning_filtered_centre")
	caspr_errors[i] = sqrt.(sum(abs2, x .- f) ./ sum(abs2, f))
	caspr_conditionings[i] = (c, cc)
end

fig, axs = plt.subplots(2, 1, figsize=(0.5PyPlotTools.latex_column, 3))
PyPlotTools.rcParams["hatch.linewidth"] = 0.0001
plt.subplots_adjust(hspace=0.3)
#axs[1].set_title("Reconstruction Error")
#axs[1].set_title("Condition Number Minimisation", y=1.1)
for i in eachindex(names_short)
	errors_before, conditionings_before, normalisation = load("../fully_sampled_criterion/" * schedules[i], "errors", "conditionings", "normalisation")
	errors_before = errors_before[:, :, 1]
	normalisation = dropdims(normalisation; dims=(1,2))
	#spread_before = compute_percentiles(errors_before, normalisation)
	errors_before = dropdims(median(sqrt.(sum(errors_before.^2; dims=1) ./ sum(normalisation)); dims=2); dims=(1, 2))
	conditionings_before = conditionings_before[:, 1]
	conditionings_before .^= 2 # Gamma_k -> M_k
	#conditioning_spread = compute_percentiles(conditionings_before)
	@show conditionings_before = dropdims(median(conditionings_before; dims=1); dims=1)
	if i == 1
		label = ("random uniform", "optimised uniform", "CASPR", "CASPR \$k\$-space centre")
	else
		label = (nothing, nothing, nothing, nothing)
	end
	@show errors = median([sqrt(sum(abs2, fullysampled_recons[i] .- lr_inversion_recons[i][j]) ./ sum(abs2, fullysampled_recons[i])) for j = 1:length(lr_inversion_recons[i])])
	axs[1].bar([i-0.2], errors_before; width=0.2, color="tab:blue", label=label[1])
	axs[1].bar([i], errors; width=0.2, color="tab:orange", label=label[2])
	axs[1].bar([i+0.2], caspr_errors; width=0.2, color="tab:green", zorder=-1, label=label[3])
	axs[2].bar([i-0.2], conditionings_before; width=0.2, color="tab:blue", label=label[1])
	axs[2].bar([i], conditionings[i][end]; width=0.2, color="tab:orange", label=label[2])
	axs[2].bar([i+0.15], [caspr_conditionings[i][1]]; width=0.1, color="tab:green", linewidth=0.2, edgecolor="black", label=label[3])
	axs[2].bar([i+0.25], [caspr_conditionings[i][2]]; width=0.1, color="tab:green", linewidth=0.2, edgecolor="black", hatch="////", label=label[4])
end
axs[1].set_xticks(1:4)
axs[1].set_xticklabels(names_short)
#axs[1].set_yscale("log")
axs[1].set_ylabel("NRMSE")
axs[2].set_yscale("log")
axs[2].set_xticks(1:4)
axs[2].set_xticklabels(names_short)
axs[2].set_ylabel("Condition Number \$\\kappa\$")
axs[2].set_ylim([1e0, 5e16])
axs[2].set_yticks([1e0, 1e8, 1e16])
axs[2].legend(loc="center", ncol=2, bbox_to_anchor=(0.45, -0.45))
plt.savefig("error_and_optimisation.eps")
plt.close(fig)
GC.gc()



# Plot reconstructions for supplementary
files = ("Jiang2014_1032_recon.jld2", "Zhao2019_400_recon.jld2", "Koolstra2018_240_recon.jld2", "FastCycle_840_recon.jld2")
fig, axs = plt.subplots(4, 4 * 4, figsize=(22, PyPlotTools.latex_column))
Δ = 20 / 4
plt.subplots_adjust(hspace=-0.1, wspace=0.05)
fig.suptitle("\\hspace*{$(-Δ * 0.0)in} Jiang \\hspace*{$(Δ*0.9)in} Zhao \\hspace*{$(Δ*0.9)in} Koolstra \\hspace*{$(Δ*0.9)in} Fast cycle", y=1.08, fontsize=20)
foreach(s -> axs[s, 1].set_ylabel("\$\\sigma_$s\$"), 1:num_σ)
for i in eachindex(names_short)
	randomuniform = load("../fully_sampled_criterion/" * files[i], "recon")
	optimiseduniform = lr_inversion_recons[i][1]
	caspr, fullysampled = load("../../../20230713_Scanning/simulation/single_channel/$(("Jiang2014", "Zhao2019", "Koolstra2018", "FastCycle")[i])_recons.jld2", "lr_inversion_recons", "fullysampled_recon")
	idx = collect(
		MRIRecon.centre_indices.(shape[1:2], (240, 210))
		for shape in (size(randomuniform), size(optimiseduniform), size(caspr))
	)
	axs[1, 4*(i-1)+1].set_title("Fully sampled")
	axs[1, 4*(i-1)+2].set_title("Random uniform")
	axs[1, 4*(i-1)+3].set_title("Optimised uniform")
	axs[1, 4*(i-1)+4].set_title("CASPR")
	@views for s = 1:num_σ
		image = axs[s, 4*(i-1)+1].imshow(abs.(fullysampled[idx[1]..., s]), origin="upper", cmap="Greys_r")
		vmin, vmax = image.get_clim()
		axs[s, 4*(i-1)+2].imshow(abs.(randomuniform[idx[1]..., s]); vmin, vmax, origin="upper", cmap="Greys_r")
		axs[s, 4*(i-1)+3].imshow(abs.(optimiseduniform[idx[2]..., s]); vmin, vmax, origin="upper", cmap="Greys_r")
		axs[s, 4*(i-1)+4].imshow(abs.(caspr[idx[3]..., s]); vmin, vmax, origin="upper", cmap="Greys_r")
		axs[s, 4*(i-1)+1].tick_params(labelleft=false, left=false, labelbottom=false, bottom=false)
		axs[s, 4*(i-1)+2].tick_params(labelleft=false, left=false, labelbottom=false, bottom=false)
		axs[s, 4*(i-1)+3].tick_params(labelleft=false, left=false, labelbottom=false, bottom=false)
		axs[s, 4*(i-1)+4].tick_params(labelleft=false, left=false, labelbottom=false, bottom=false)
	end
	GC.gc()
end
plt.savefig("recons.eps")
plt.close(fig)
GC.gc()

