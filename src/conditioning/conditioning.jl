using Revise
using JLD2
using ImageView
using FFTW
using IterativeSolvers
using Statistics
import MRIConst
import MRFingerprinting as MRF
import MRFSeqParams
import XLargerYs
import MRIPhantoms
import MRIRecon
import MRITrajectories
using PlasticArrays
import IterativeSolverTools
using LinearAlgebra
import PyPlot as plt 
import PyPlotTools
PyPlotTools.set_image()
# TODO: brainweb, also check dependencies
include("../../../20230324_BrainWebTest/load_brainweb.jl")
BLAS.set_num_threads(Threads.nthreads())
FFTW.set_num_threads(Threads.nthreads())

# Construct phantom
num_σ = 4
phantom_fractions = load_brainweb("../../../20230324_BrainWebTest/brainweb.jld2")
phantom_fractions[:, :, size(phantom_fractions, 3) ÷ 2, :];
#sagittal: phantom_fractions = permutedims(phantom_fractions[:, size(phantom_fractions, 2) ÷ 2+30, end:-1:1, :], (2, 1, 3))
upsampled_shape = size(phantom_fractions)[1:2]
shape = (num_columns, num_lines) = upsampled_shape .÷ 2
num_tissues = size(phantom_fractions, 3)
relB1_tissue = ones(num_tissues)
R1 = 1 ./ MRIConst.brainweb_parameters["T1"]
R2 = 1 ./ MRIConst.brainweb_parameters["T2"]
R_tissue = [(r2, r1) for (r2, r1) in zip(R2, R1)]

function compute_conditioning(VH, first_sample)
	num_time = size(VH, 2)
	conditioning = Vector{Vector{Float64}}(undef, 3)
	times = Vector{Int}(undef, 4)
	times[1] = first_sample
	for i = 1:3
		conditioning[i] = Vector{Float64}(undef, num_time)
		for t = 1:num_time
			conditioning[i][t] = cond(VH[:, [times[1:i]..., t]])
		end
		times[i+1] = argmin(round.(conditioning[i], digits=6))
	end
	return conditioning, times 
end

schedules = (
	"../schedules/Jiang2014/compressed_dictionary.jld2",
	"../schedules/Zhao2019/compressed_dictionary.jld2", 
	"../schedules/Koolstra2018/compressed_dictionary.jld2",
	"../schedules/FastCycle/compressed_dictionary.jld2",
	"../schedules/Fourier/Fourier.jld2"
)
first_sample = 20
titles = (
	"\\textbf{Jiang}", 
	"\\textbf{Zhao}",
	"\\textbf{Koolstra}",
	"\\textbf{Fast cycle}",
	"\\textbf{Fourier}"
)
title_addition = ("second", "third", "fourth")
xticks = (
	[1, 500, 1000],
	[1, 200, 400],
	[1, 100, 240],
	[1, 400, 840],
	[1, 450, 900]
)
t0s = (2, 2, 2, 1, 1)


fig, axs = plt.subplots(3, 5; sharey=true, sharex="col", figsize=(1.2 * PyPlotTools.latex_column, 2))
plt.subplots_adjust(left=0, right=1, wspace=0.2, hspace=0.7)
for (i, (path, title, xticks, t0)) in enumerate(zip(schedules, titles, xticks, t0s))
	VH = load(path, "transform")[1:num_σ, t0:end]
	conditioning, times = compute_conditioning(VH, first_sample)
	num_time = length(conditioning[1])
	for j = 1:3
		axs[j, i].set_ylim([0, 1])
		axs[j, i].plot(1:num_time, 1 ./ conditioning[j]; color="tab:blue", zorder=1)
		for k = 1:j
			axs[j, i].axvline(times[k]; zorder=0, color="tab:orange", linestyle="dotted", linewidth=0.75)
		end
		axs[j, i].axvline(times[j+1]; zorder=0, color="tab:green", linestyle="dashed", linewidth=0.75)
		axs[j, i].set_xticks(xticks)
		if j == 1
			axs[j, i].set_title(title * "\n$(title_addition[j]) sample"; fontdict=Dict("fontsize"=>10))
		else
			axs[j, i].set_title("$(title_addition[j]) sample"; fontdict=Dict("fontsize"=>10))
		end
		if j == 3
			axs[j, i].set_xlabel("\$t\$ [index]"; fontdict=Dict("fontsize"=>10))
		end
		axs[j, 1].set_ylabel("\$1 / \\kappa_$(j+1)\$")
	end
end
axs[1, 1].set_yticks([0, 0.5, 1.0])
plt.savefig("conditioning_sampling.eps")
plt.close(fig)

