using Revise
using JLD2
using Statistics
import MRFingerprinting as MRF
import PyPlot as plt
import PyPlotTools
PyPlotTools.set_image()
num_σ = 4

function compute_percentiles(errors, normalisation)
	hcat(
		[
			quantile(
				dropdims(sqrt.(sum(errors[:, :, i].^2; dims=1) ./ sum(normalisation)); dims=1),
				0.2
			) for i in 1:length(kspaces_iter)
		],
		[
			quantile(
				dropdims(sqrt.(sum(errors[:, :, i].^2; dims=1) ./ sum(normalisation)); dims=1),
				0.8
			) for i in 1:length(kspaces_iter)
		]
	)'
end
function compute_percentiles(conditioning)
	hcat(
		[
			quantile(
				conditioning[:, i],
				0.2
			) for i in 1:length(kspaces_iter)
		],
		[
			quantile(
				conditioning[:, i],
				0.8
			) for i in 1:length(kspaces_iter)
		]
	)'
end

schedules = ("Jiang2014_1032_recons_iter.jld2", "Zhao2019_400_recons_iter.jld2", "Koolstra2018_240_recons_iter.jld2", "FastCycle_840_recons_iter.jld2")
num_points = 7
kspaces_iter = load(schedules[1], "kspaces_to_sample")[1:num_points]

fig, axs = plt.subplots(2, 1; figsize=(0.45PyPlotTools.latex_column, 3.5))
#axs[1].set_title("Reconstruction Error")
capsize=1.25
signs = ones(2, num_points)
signs[1, :] .= -1 # matplotlib argggg
offsets = (-0.15, -0.05, 0.05, 0.15)
labels = ("Jiang", "Zhao", "Koolstra", "FastCycle")
styles = ("-", "dashed", "dotted", "dashdot")
colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
axs[1].set_ylabel("NRMSE [\\SI{}{\\percent}]")
axs[1].set_xticks(kspaces_iter)
axs[1].set_xlabel("Number of measured \$k\$-spaces \$K\$")
#axs[2].set_title("Conditioning of the System Matrix")
axs[2].set_yscale("log")
axs[2].set_ylabel("\$\\kappa\$")
axs[2].set_xlabel("Number of measured \$k\$-spaces \$K\$")
axs[2].set_xticks(kspaces_iter)
for (i, path) in enumerate(schedules)
	errors, conditionings, normalisation = load(path, "errors", "conditionings", "normalisation")
	errors        = errors[:, :, 1:num_points]
	conditionings = conditionings[:, 1:num_points]
	normalisation = dropdims(normalisation; dims=(1,2))
	spread        = compute_percentiles(errors, normalisation)
	errors        = dropdims(median(sqrt.(sum(errors.^2; dims=1) ./ sum(normalisation)); dims=2); dims=(1, 2))
	@show errors[end]
	conditionings .^= 2 # Gamma_k -> M_k
	conditioning_spread = compute_percentiles(conditionings)
	conditioning  = dropdims(median(conditionings; dims=1); dims=1)
	@show conditioning[end]
	@views axs[1].errorbar(
		x=kspaces_iter .+ offsets[i],
		y=100 .* errors,
		yerr=100 .* signs .* (spread .- errors');
		marker="o", capsize, linestyle=styles[i], color=colors[i], linewidth=0.5, label=labels[i]
	)
	axs[2].errorbar(
		x=kspaces_iter .+ offsets[i],
		y=conditioning,
		yerr=signs .* (conditioning_spread .- conditioning');
		marker="o", capsize, linestyle=styles[i], color=colors[i], linewidth=0.5, label=labels[i]
	)
end
axs[1].legend(handlelength=1.5, ncol=2)
axs[2].legend(ncol=2)
plt.subplots_adjust(hspace=0.45)
plt.savefig("iterative_error.eps")
plt.close(fig)

