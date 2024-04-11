using Revise
using Random
using JLD2
#using ImageView
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
phantom_fractions = load_brainweb("../../../20230324_BrainWebTest/brainweb.jld2")
phantom_fractions = phantom_fractions[:, :, size(phantom_fractions, 3) ÷ 2, :];
upsampled_shape = size(phantom_fractions)[1:2]
shape = upsampled_shape .÷ 2
num_tissues = size(phantom_fractions, 3)
num_σ = 4

# Get dictionary and parameters
dictionary_paths = (
	("Jiang2014", "../schedules/Jiang2014", 1032, (240, 215)),
	("Zhao2019", "../schedules/Zhao2019", 400, (240, 240)),
	("Koolstra2018", "../schedules/Koolstra2018", 240, (240, 240)),
	("FastCycle", "../schedules/FastCycle", 840, (240, 210)),
)

function optimise_sampling(dictionary_path, shape, num_time)
	VH = load(joinpath(dictionary_path, "compressed_dictionary.jld2"), "transform")[1:num_σ, 1:num_time]
	#num_time = size(VH, 2)
	num_kspaces = num_σ
	num_samples = num_kspaces * prod(shape)
	num_threads = Threads.nthreads()
	best_conditioning = Inf
	local best_samplings::Array{Vector{CartesianIndex{2}}}
	local best_conditionings::Array{Float64, 1}
	local best_iterations::Array{Int, 1}
	L = ReentrantLock()
	#active_threads = Base.Semaphore(num_threads)
	@sync for _ = 1:num_threads
		Threads.@spawn begin
			for q = 1:5
				samplings = Array{Vector{CartesianIndex{2}}}(undef, 0)
				conditionings = Array{Float64}(undef, 0)
				iterations = Array{Int}(undef, 0)
				#split_sampling = MRITrajectories.distribute_dynamically(
				#	MRITrajectories.rand(Int, () -> (rand(), rand()), num_samples, shape, num_σ, num_time; maxiter=10_000_000),
				#	num_time
				#)
				split_sampling = MRIRecon.split_sampling(
					MRITrajectories.regular_dynamic(shape, num_time, num_kspaces),
					#MRITrajectories.uniform_dynamic(shape, num_time, ceil(Int, num_samples / num_time));
					num_time
				)
				#
				swapped = 0
				maxiter = 1000
				split_sampling, conditioning, swapped = MRF.improve_sampling_3!(split_sampling, VH, shape, 1; maxiter)
				push!(iterations, swapped)
				push!(conditionings, conditioning)
				sampling_indices = MRIRecon.in_chronological_order(split_sampling)
				push!(samplings, sampling_indices)
				while swapped > 0
					split_sampling, conditioning, swapped = MRF.improve_sampling_3!(
						split_sampling,
						VH,
						shape,
						maxiter;
						maxiter
					)
					#split_sampling, conditioning, swapped = MRF.improve_sampling!(
					#	split_sampling,
					#	VH,
					#	shape,
					#	iter
					#)
					#println("$tid, $n, $conditioning")
					#if i < opt_iterations
					#	error("Can't optimise further")
					#end
					@show Threads.threadid(), swapped, conditioning
					#flush(stdout)
					push!(iterations, swapped)
					push!(conditionings, conditioning)
					sampling_indices = MRIRecon.in_chronological_order(split_sampling)
					push!(samplings, sampling_indices)
					@assert length(sampling_indices) == num_samples
					@assert MRIRecon.is_unique_sampling(sampling_indices, num_time)
				end
				conditioning = conditionings[end]
				lock(L)
				if conditioning < best_conditioning
					best_conditioning = conditioning
					best_conditionings = conditionings
					best_iterations = iterations
					best_samplings = samplings
					println("best conditioning $best_conditioning")
					flush(stdout)
				end
				unlock(L)
				#Base.release(active_threads)
			end
		end
	end
	return best_samplings, best_conditionings, best_iterations
end
for (name, p, num_time, shape) in dictionary_paths
	best_samplings, best_conditionings, best_iterations = optimise_sampling(p, shape, num_time)
	@show best_iterations
	@show best_conditionings
	jldsave("$(name)_optimised_sampling_long.jld2"; best_samplings, best_conditionings, best_iterations)
end

# Recon

function run(dictionary_path, samplings, upsampled_shape, shape)
	D = load(joinpath(dictionary_path, "compressed_dictionary.jld2"), "compressed_dictionary")[:, 1:num_σ]
	VH = load(joinpath(dictionary_path, "compressed_dictionary.jld2"), "transform")[1:num_σ, :]
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
	num_runs = length(samplings)
	lr_inversion_recons = Array{Array{ComplexF64, 3}}(undef, num_runs)
	infos = Array{NTuple{3, Vector{Float64}}}(undef, num_runs)
	for n = 1:num_runs
		@show n
		lr_inversion_recons[n], infos[n] = reconstruct(samplings[n])
	end
	return lr_inversion_recons, fullysampled_recon, infos
end
for (name, p) in dictionary_paths
	samplings, conditionings, iterations = load(joinpath(p, "$(name)_optimised_sampling.jld2"), "best_samplings", "best_conditionings", "best_iterations")
	lr_inversion_recons, fullysampled_recon, infos = run(dictionary_sine..., samplings, upsampled_shape, shape);
	jldsave("$(name)_recons.jld2"; lr_inversion_recons, fullysampled_recon, infos)
end

