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
import MRFSeqParams
using PlasticArrays
using LinearAlgebra
BLAS.set_num_threads(1)

function optimise_sampling(VH)
	@show num_σ, num_time = size(VH)
	@show num_patterns = num_time ÷ num_σ
	@assert num_patterns * num_σ == num_time
	num_kspaces = num_σ
	num_threads = Threads.nthreads()
	best_conditioning = Inf
	local best_samplings::Array{Vector{CartesianIndex{2}}}
	local best_conditionings::Array{Float64, 1}
	local best_iterations::Array{Int, 1}
	L = ReentrantLock()
	@sync for _ = 1:num_threads
		Threads.@spawn begin
			for q = 1:5
				#@show "new run"
				samplings = Array{Vector{CartesianIndex{2}}}(undef, 0)
				conditionings = Array{Float64}(undef, 0)
				iterations = Array{Int}(undef, 0)
				#
				sampling_indices = shuffle!([CartesianIndex((i-1) ÷ num_σ + 1, 1) for i in 1:num_time])
				split_sampling = MRIRecon.split_sampling(sampling_indices, num_time)
				#
				maxiter = 1000
				swapped = 0
				split_sampling, conditioning, swapped = MRF.improve_sampling_3!(
					split_sampling, VH, (num_patterns, 1), 1000;
					maxiter
				)
				push!(iterations, swapped)
				push!(conditionings, conditioning)
				sampling_indices = MRIRecon.in_chronological_order(split_sampling)
				push!(samplings, sampling_indices)
				already_was_one = 0
				while swapped > 0
					#@show "before", Threads.threadid(), conditioning
					split_sampling, conditioning, swapped = MRF.improve_sampling_3!(
						split_sampling,
						VH,
						(num_patterns, 1),
						1000;
						maxiter
					)
					#@show "after", Threads.threadid(), conditioning
					if swapped == 1
						already_was_one += 1
						if already_was_one == 10
							break
						end
					end
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
VH_Jiang = load("../schedules/Jiang2014/compressed_dictionary.jld2", "transform")[1:num_σ, 2:end];
VH_Zhao = load("../schedules/Zhao2019/compressed_dictionary.jld2", "transform")[1:num_σ, 2:end];
VH_Koolstra = load("../schedules/Koolstra2018/compressed_dictionary.jld2", "transform")[1:num_σ, 2:end];
VH_FastCycle = load("../schedules/FastCycle/compressed_dictionary.jld2", "transform")[1:num_σ, 1:end];
#= Inversion pulse included for Fast cycle, because signal is non-zero after inversion because driven equilibrium.
# For the other schedules, starting from thermal equilibrium, the inversion gives zero signal and must hence
be excluded =#

best_samplings_Jiang, best_conditionings_Jiang, best_iterations_Jiang = optimise_sampling(VH_Jiang[1:num_σ, :])
best_samplings_Zhao, best_conditionings_Zhao, best_iterations_Zhao = optimise_sampling(VH_Zhao[1:num_σ, :])
best_samplings_Koolstra, best_conditionings_Koolstra, best_iterations_Koolstra = optimise_sampling(VH_Koolstra[1:num_σ, :])
best_samplings_FastCycle, best_conditionings_FastCycle, best_iterations_FastCycle = optimise_sampling(VH_FastCycle[1:num_σ, :])

jldsave("optimised_sampling.jld2";
	num_σ,
	best_samplings_Jiang, best_conditionings_Jiang, best_iterations_Jiang,
	best_samplings_Zhao, best_conditionings_Zhao, best_iterations_Zhao,
	best_samplings_FastCycle, best_conditionings_FastCycle, best_iterations_FastCycle,
	best_samplings_Koolstra, best_conditionings_Koolstra, best_iterations_Koolstra
)

