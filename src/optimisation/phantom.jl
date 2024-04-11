using Revise
using Random
using JLD2
using Primes
using ImageView
using FFTW
using IterativeSolvers
using Statistics
#import PyPlot as plt 
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
include("../../../20230324_BrainWebTest/load_brainweb.jl")
BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

# Construct phantom
phantom_fractions = load_brainweb("../../../20230324_BrainWebTest/brainweb.jld2");
phantom_fractions = phantom_fractions[:, :, size(phantom_fractions, 3) ÷ 2, :]
upsampled_shape = size(phantom_fractions)[1:2]
shape = upsampled_shape .÷ 2
num_tissues = size(phantom_fractions, 3)
num_σ = 4

# Get dictionary and parameters
dictionary_paths = (
	("Jiang2014", "../schedules/Jiang2014", 215, 2),
	("Zhao2019", "../schedules/Zhao2019", 240, 2),
	("Koolstra2018", "../schedules/Koolstra2018", 240, 2),
	("FastCycle", "../schedules/FastCycle", 210, 1)
)

function extend_sampling(sampling, num_σ, shape)
	num_time = length(sampling)
	blocksize = num_time ÷ num_σ
	indices = shuffle(CartesianIndices(shape))
	extended_sampling = Vector{CartesianIndex{2}}(undef, length(indices) * num_σ)
	k = 1
	frequency = zeros(Integer, length(indices))
	for i in 1:blocksize:length(indices)
		for j = 1:num_time
			extended_sampling[k] = indices[i-1 + mod1(sampling[j][1], length(indices)-i+1)]
			frequency[i-1 + mod1(sampling[j][1], length(indices)-i+1)] += 1
			k += 1
			k > length(extended_sampling) && break
		end
	end
	@show allequal(frequency), frequency[1]
	return extended_sampling
end

samplings = [
	load("optimised_sampling.jld2", "best_samplings_$(name)")[[1, end]]
	for name in ("Jiang", "Zhao", "Koolstra", "FastCycle")
]

function run(dictionary_path, samplings, phantom_fractions, upsampled_shape, shape, t0)
	D = load(joinpath(dictionary_path, "compressed_dictionary.jld2"), "compressed_dictionary")[:, 1:num_σ]
	VH = load(joinpath(dictionary_path, "compressed_dictionary.jld2"), "transform")[1:num_σ, t0:end]
	R, relB1 = load(joinpath(dictionary_path, "parameters.jld2"), "R", "relB1");
	R = XLargerYs.combinations(R);
	relB1_tissue = ones(num_tissues)
	R1 = 1 ./ MRIConst.brainweb_parameters["T1"]
	R2 = 1 ./ MRIConst.brainweb_parameters["T2"]
	R_tissue = [(r2, r1) for (r2, r1) in zip(R2, R1)]
	phantom_fingerprints = MRIConst.brainweb_parameters["PD"] .* D[MRF.closest((R_tissue, relB1_tissue), (R, relB1)), :]
	#
	V_conj, VT = MRF.convenient_Vs(VH)
	num_time = size(VH, 2)
	#
	cut = div(size(phantom_fractions, 2) - upsampled_shape[2], 2)
	phantom_fractions = phantom_fractions[:, cut+1:size(phantom_fractions, 2)-cut, :]
	new_samplings = Vector{Vector{CartesianIndex{2}}}(undef, 20)
	for i in eachindex(new_samplings)
		#j = mod1(i, length(samplings))
		if length(samplings[end]) < num_σ * prod(shape)
			new_samplings[i] = extend_sampling(samplings[end], num_σ, shape)
		else
			new_samplings[i] = samplings[end]
		end
	end
	samplings = new_samplings
	#begin
	#	λ = MRF.compute_eigenvalues(VH, shape, MRIRecon.split_sampling_spatially(samplings[1], shape, num_time))
	#	@show MRF.calculate_conditioning(λ)
	#end
	#imshow(sum(dropdims(MRIRecon.sampling_mask(samplings[1], size(phantom_fractions)[1:2] .÷ 2, num_time); dims=1); dims=3))
	#@show size(phantom_fractions), num_time, length.(samplings)
	highres_phantom = reshape(reshape(phantom_fractions, :, num_tissues) * phantom_fingerprints, upsampled_shape..., num_σ)
	num_channels = 1
	kspace = MRIPhantoms.measure(highres_phantom, (2, 2), shape) # Is already correctly fftshifted
	fullysampled_recon = ifft(kspace, 1:2) # Already shifted correctly
	#
	function reconstruct(sampling_indices)
		# Fourier operator
		F = MRIRecon.plan_fourier_transform(Array{ComplexF64}(undef, shape..., num_channels * num_σ), 1:2)
		# Sampling
		!MRIRecon.is_unique_sampling(sampling_indices, num_time) && println("Sampling not unique")
		sampling_indices = fftshift(sampling_indices, shape)
		# Masks and lr-mixing
		lr_mix = MRF.lowrank_mixing(VH, sampling_indices, shape)
		#
		M = MRF.plan_lowrank_mixing(lr_mix, 1, num_channels)
		A = F' * M * F
		GC.gc(true)
		#
		# Run reconstruction
		@views begin
			backprojection = copy(F' * M * vec(kspace))
			lr_inversion_recon = copy(backprojection)
			_, residuals, A_norm, nrmse = IterativeSolverTools.debug_cg!(
				lr_inversion_recon,
				A,
				backprojection,
				vec(fullysampled_recon);
				maxiter=50
			)
		end
		return reshape(lr_inversion_recon, shape..., num_σ), (residuals, A_norm, nrmse)
	end
	#
	@show num_runs = length(samplings)
	lr_inversion_recons = Array{Array{ComplexF64, 3}}(undef, num_runs)
	infos = Array{NTuple{3, Vector{Float64}}}(undef, num_runs)
	Threads.@threads for n = 1:num_runs
		@show n
		sampling = samplings[n]
		lr_inversion_recons[n], infos[n] = reconstruct(sampling)
	end
	return lr_inversion_recons, fullysampled_recon, infos
end

for ((name, path, cut, t0), sampling) in zip(dictionary_paths, samplings)
	cut_shape = (shape[1], cut)
	lr_inversion_recons, fullysampled_recon, infos = run(path, sampling, phantom_fractions, 2 .* cut_shape, cut_shape, t0);
	jldsave("$(name)_recons.jld2"; lr_inversion_recons, fullysampled_recon, infos)
end

