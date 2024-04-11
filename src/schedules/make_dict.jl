using Revise
using JLD2
using XLargerYs
using ThreadTools
import MRIEPG
import MRFSeqParams
import MRFingerprinting
import PyPlot as plt
import PyPlotTools
using LinearAlgebra

function make_dictionary(
	constTR::Val{true},
	cycles::Integer,
	relB1::AbstractVector{<: Real},
	R::XLargerY{<: Real},
	α::AbstractVector{<: Real},
	TR::AbstractVector{<: Real},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)
	timepoints_per_cycle = length(α)
	timepoints = cycles * timepoints_per_cycle
	ϕ = zeros(timepoints_per_cycle)
	# Compute relaxation
	@views relaxation_inversion, _ = MRIEPG.compute_relaxation(
		TR[1], R, G,
		[τ[1:end-1]; TR[1] - sum(τ[1:end-1])],
		D, kmax
	)
	@assert all(isequal(TR[2]), @view TR[3:end-1])
	relaxation, num_systems = MRIEPG.compute_relaxation(TR[2], R, G, τ, D, kmax)
	@views relaxation_waiting, _ = MRIEPG.compute_relaxation(
		TR[end], R, G,
		[τ[1:end-1]; TR[end] - sum(τ[1:end-1])],
		D, kmax
	)
	# Allocate memory
	dictionary = Array{Float64, 3}(undef, R.N, length(α), length(relB1))
	# Inversion pulse, don't scale adiabatic pulse so can precompute it
	rf_matrix_inversion = Array{ComplexF64, 3}(undef, 3, 3, 1)
	@views MRIEPG.rf_pulse_matrix!(rf_matrix_inversion[:, :, 1], α[1], ϕ[1])
	num_threads = Threads.nthreads()
	# Constant TR rf matrices
	rf_matrices = [
		Array{ComplexF64, 3}(undef, 3, 3, timepoints_per_cycle-2)
		for _ = 1:num_threads
	]
	# Pulse for last TR, depends on thread
	rf_matrix_waiting = [
		Array{ComplexF64, 3}(undef, 3, 3, 1)
		for _ = 1:num_threads
	]
	# Simulate
	memory = [
		MRIEPG.allocate_memory(
			Val(:minimal),
			timepoints, num_systems, kmax,
			nothing,
			Val(:signal)
		)
		for _ = 1:Threads.nthreads()
	]
	function run!(cycle::Integer, memory::MRIEPG.SimulationMemory{Matrix{ComplexF64}, Matrix{ComplexF64}})
		t = (cycle-1) * timepoints_per_cycle + 1
		memory = MRIEPG.simulate!(
			t,
			timepoints,
			rf_matrix_inversion,
			relaxation_inversion,
			num_systems,
			kmax, Val(:minimal),
			memory
		)
		memory = MRIEPG.simulate!(
			t+1,
			timepoints,
			rf_matrices[Threads.threadid()],
			relaxation,
			num_systems,
			kmax, Val(:minimal),
			memory
		)
		memory = MRIEPG.simulate!(
			t+timepoints_per_cycle-1,
			timepoints,
			rf_matrix_waiting[Threads.threadid()],
			relaxation_waiting,
			num_systems,
			kmax, Val(:minimal),
			memory
		)
		return memory
	end
	Threads.@threads :static for i in eachindex(relB1)
		tid = Threads.threadid()
		thread_rf_matrices = rf_matrices[tid]
		thread_rf_matrix_waiting = rf_matrix_waiting[tid]
		thread_memory = memory[tid]
		# Compute pulse matrices
		for t = 2:timepoints_per_cycle-1
			@views MRIEPG.rf_pulse_matrix!(thread_rf_matrices[:, :, t-1], relB1[i] * α[t], ϕ[t])
		end
		@views MRIEPG.rf_pulse_matrix!(thread_rf_matrix_waiting[:, :, 1], relB1[i] * α[end], ϕ[end])
		# Run
		thread_memory = MRIEPG.driven_equilibrium!(
			cycles,
			run!,
			thread_memory,
			Val(:signal)
		)
		# Reset initial state in memory, in the first iteration, it is read from memory.two_states[2]
		MRIEPG.reset_source_state!(thread_memory, num_systems)
		MRIEPG.reset_target_state!(thread_memory)
		memory[tid] = thread_memory
		# Record
		@views @. dictionary[:, :, i] = imag(thread_memory.recording[:, end-timepoints_per_cycle+1:end])
		# Note: this allocates since dictionary is not zeros... it's a system thing
	end
	return dictionary
end

function make_dictionary(
	constTR::Val{false},
	cycles::Integer,
	relB1::AbstractVector{<: Real},
	R::XLargerY{<: Real},
	α::AbstractVector{<: Real},
	TR::AbstractVector{<: Real},
	G::AbstractVector{<: Real},
	τ::AbstractVector{<: Real},
	D::Real,
	kmax::Integer
)
	timepoints_per_cycle = length(α)
	timepoints = cycles * timepoints_per_cycle
	ϕ = zeros(timepoints_per_cycle)
	# Compute relaxation
	relaxation, num_systems = MRIEPG.compute_relaxation(TR, R, G, τ, D, kmax)
	# Allocate memory
	dictionary = Array{Float64, 3}(undef, R.N, length(α), length(relB1))
	# Inversion pulse, don't scale adiabatic pulse so can precompute it
	num_threads = Threads.nthreads()
	# Constant TR rf matrices
	rf_matrices = [
		Array{ComplexF64, 3}(undef, 3, 3, timepoints_per_cycle)
		for _ = 1:num_threads
	]
	for tid = 1:num_threads
		@views MRIEPG.rf_pulse_matrix!(rf_matrices[tid][:, :, 1], α[1], ϕ[1])
	end
	# Simulate
	memory = [
		MRIEPG.allocate_memory(
			Val(:minimal),
			timepoints, num_systems, kmax,
			nothing,
			Val(:signal)
		)
		for _ = 1:Threads.nthreads()
	]
	function run!(cycle::Integer, memory::MRIEPG.SimulationMemory{Matrix{ComplexF64}, Matrix{ComplexF64}})
		t = (cycle-1) * timepoints_per_cycle + 1
		memory = MRIEPG.simulate!(
			t,
			timepoints,
			rf_matrices[Threads.threadid()],
			relaxation,
			num_systems,
			kmax, Val(:minimal),
			memory
		)
		return memory
	end
	Threads.@threads :static for i in eachindex(relB1)
		tid = Threads.threadid()
		thread_rf_matrices = rf_matrices[tid]
		thread_memory = memory[tid]
		# Compute pulse matrices
		for t = 2:timepoints_per_cycle
			@views MRIEPG.rf_pulse_matrix!(thread_rf_matrices[:, :, t], relB1[i] * α[t], ϕ[t])
		end
		# Run
		thread_memory = MRIEPG.driven_equilibrium!(cycles, run!, thread_memory, Val(:signal))
		# Reset initial state in memory, in the first iteration, it is read from memory.two_states[2]
		MRIEPG.reset_source_state!(thread_memory, num_systems)
		MRIEPG.reset_target_state!(thread_memory)
		memory[tid] = thread_memory
		# Record
		@views @. dictionary[:, :, i] = imag(thread_memory.recording[:, end-timepoints_per_cycle+1:end])
		# Note: this allocates since dictionary is not zeros... it's a system thing
	end
	return dictionary
end

function make_dict(out, cycles, do_relB1, constTR)
	# Get parameters
	# Pulse asymmetries don't matter here because relevant pulses have the same echo echo time
	TR, α = MRFSeqParams.read_pulses(joinpath(out, "MRFParams"))[6:7]
	#@show TR, α
	#return
	# Gradients and diffusion
	kmax = 50
	#@show cycles = max(ceil(Int, 2 * 5000 / sum(TR)), 2)
	G = Float64[0, 12, 5, 42, 0]
	τ = [1.2, 0.4, 2.0, 0.5, 0.9]
	D = 2e-12
	#
	# Set up tissue parameters
	T1 = [50:10:1200; 1250:50:2000; 2100:100:3000; 3100:300:4000]
	T2 = [10:2:150; 155:5:250; 260:10:400; 450:50:1000]
	R = XLargerY(1 ./ reverse(T2), 1 ./ reverse(T1))
	if do_relB1
		relB1 = range(0.25, 1.75; length=101)[1:end-1]
	else
		relB1 = [1.0]
	end
	jldsave(joinpath(out, "parameters.jld2"); relB1, R)
	num_σ_cutoff = 100
	println("Number of atoms $(R.N * length(relB1)) and approx. size $(round(R.N * length(relB1) * min(length(α), num_σ_cutoff) * 8 * 1e-9; digits=2))Gb")
	#
	BLAS.set_num_threads(1) # Parallelisation happens on another level
	dictionary = make_dictionary(constTR, cycles, relB1, R, α, TR, G, τ, D, kmax);
	BLAS.set_num_threads(Threads.nthreads())
	#
	# Reshape
	dictionary = reshape(permutedims(dictionary,  (1, 3, 2)), :, length(α))
	#
	# Normalise and compress for matching
	normalised_dictionary, norms = MRFingerprinting.normalise(dictionary)
	GC.gc(true)
	compressed_dictionary, singular_values, transform = MRFingerprinting.compress(normalised_dictionary)
	@show size(compressed_dictionary)
	GC.gc(true)
	#
	# Save
	jldsave(
		joinpath(out, "compressed_dictionary.jld2");
		norms,
		compressed_dictionary=compressed_dictionary[:, 1:num_σ_cutoff],
		singular_values,
		transform
	)
	return norms, compressed_dictionary[:, 1:num_σ_cutoff], singular_values, transform
end

norms, compressed_dictionary, singular_values, transform = make_dict("Jiang2014", 1, false, Val(false))
norms, compressed_dictionary, singular_values, transform = make_dict("Zhao2019", 1, false, Val(false))
norms, compressed_dictionary, singular_values, transform = make_dict("Koolstra2018", 1, false, Val(true))
norms, compressed_dictionary, singular_values, transform = make_dict("FastCycle", 4, true, Val(true))

