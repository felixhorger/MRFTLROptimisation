
using Revise
using Random
using JLD2
using Statistics
import PyPlot as plt 
import MRIConst
import MRFingerprinting as MRF
import MRFSeqParams
import MRIRecon
import MRITrajectories
using PlasticArrays
using LinearAlgebra
BLAS.set_num_threads(1)

function optimise_sampling(VH)
	num_σ, num_time = size(VH)
	@show num_patterns = num_time ÷ num_σ
	num_kspaces = num_σ
	num_threads = Threads.nthreads()
	best_conditioning = Inf
	local best_samplings::Array{Vector{CartesianIndex{2}}}
	local best_conditionings::Array{Float64, 1}
	local best_iterations::Array{Int, 1}
	L = ReentrantLock()
	@sync for _ = 1:num_threads
		Threads.@spawn begin
			for q = 1:1
				samplings = Array{Vector{CartesianIndex{2}}}(undef, 0)
				conditionings = Array{Float64}(undef, 0)
				iterations = Array{Int}(undef, 0)
				#
				sampling_indices = [CartesianIndex((i-1) ÷ num_σ + 1, 1) for i in 1:num_time]
				split_sampling = MRIRecon.split_sampling(sampling_indices, num_time)
				#
				swapped = 1
				split_sampling, conditioning, swapped = MRF.improve_sampling_3!(split_sampling, VH, (num_patterns, 1), 1)
				push!(iterations, swapped)
				push!(conditionings, conditioning)
				sampling_indices = MRIRecon.in_chronological_order(split_sampling)
				push!(samplings, sampling_indices)
				while swapped > 0
					split_sampling, conditioning, swapped = MRF.improve_sampling_3!(
						split_sampling,
						VH,
						(num_patterns, 1),
						25_000
					)
					push!(iterations, swapped)
					push!(conditionings, conditioning)
					sampling_indices = MRIRecon.in_chronological_order(split_sampling)
					push!(samplings, sampling_indices)
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
			end
		end
	end
	return best_samplings, best_conditionings, best_iterations
end

num_σ = 4
VH = load("compressed_dictionary.jld2", "transform")[1:num_σ, :]

best_samplings, best_conditionings, best_iterations = optimise_sampling(VH)

jldsave("sampling.jld2"; num_σ, samplings=best_samplings, conditionings=best_conditionings, iterations=best_iterations)

