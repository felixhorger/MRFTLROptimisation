using Revise
using JLD2
using ImageView
using FFTW
using IterativeSolvers
import PyPlot as plt 
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
# TODO: this needs to go in
include("../../../20230324_BrainWebTest/load_brainweb.jl")
BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

# Construct phantom
# TODO: the brainweb needs to go into this
phantom_fractions = load_brainweb("../../../20230324_BrainWebTest/brainweb.jld2")
phantom_fractions = phantom_fractions[:, :, size(phantom_fractions, 3) ÷ 2, :];
upsampled_shape = size(phantom_fractions)[1:2]
num_tissues = size(phantom_fractions, 3)
relB1_tissue = ones(num_tissues)
R1 = 1 ./ MRIConst.brainweb_parameters["T1"]
R2 = 1 ./ MRIConst.brainweb_parameters["T2"]
R_tissue = [(r2, r1) for (r2, r1) in zip(R2, R1)]

dictionary_paths = (
	("Jiang2014", "../schedules/Jiang2014", 2),
	("Zhao2019", "../schedules/Zhao2019", 2),
	("Koolstra2018", "../schedules/Koolstra2018", 2),
	("FastCycle", "../schedules/FastCycle", 1)
)


function simulate(name, dictionary_path, t0)
	# Dictionary
	num_σ = 4
	dictionary = load(joinpath(dictionary_path, "compressed_dictionary.jld2"), "compressed_dictionary")[:, 1:num_σ];
	VH = load(joinpath(dictionary_path, "compressed_dictionary.jld2"), "transform")[1:num_σ, t0:end];
	num_time = size(VH, 2)
	V_conj, VT = MRF.convenient_Vs(VH)
	# Construct phantom
	R, relB1 = load(joinpath(dictionary_path, "parameters.jld2"), "R", "relB1");
	R = XLargerYs.combinations(R);
	highres_phantom = MRIConst.brainweb_parameters["PD"] .* dictionary[MRF.closest((R_tissue,), (R,)), :]
	highres_phantom = reshape(reshape(phantom_fractions, :, num_tissues) * highres_phantom, upsampled_shape..., num_σ)
	shape = (num_columns, num_lines) = upsampled_shape .÷ 2
	num_channels = 1
	kspace = MRIPhantoms.measure(highres_phantom, (2, 2), shape) # Is already correctly fftshifted
	fullysampled = ifft(kspace, 1:2) # Already shifted correctly
	# Function to reconstruct
	function reconstruct(sampling_indices)
		# Sampling
		!MRIRecon.is_unique_sampling(sampling_indices, num_time) && println("Sampling not unique")
		sampling_indices = fftshift(sampling_indices, shape)
		# LR-mixing
		lr_mix = MRF.lowrank_mixing(VH, sampling_indices, shape)
		M = MRF.plan_lowrank_mixing(lr_mix, 1, num_channels)
		# Fourier
		F = MRIRecon.plan_fourier_transform(Array{ComplexF64}(undef, num_columns, num_lines, num_channels * num_σ), 1:2)
		# System matrix
		A = F' * M * F
		#
		GC.gc(true)
		#
		# Reconstruction
		maxiter = 50
		backprojection = copy(F' * M * vec(kspace))
		local lr_inversion_recon = copy(backprojection)
		_, residuals, A_norm, nrmse = IterativeSolverTools.debug_cg!(lr_inversion_recon, A, backprojection, vec(fullysampled); maxiter)
		#
		return reshape(lr_inversion_recon, num_columns, num_lines, num_σ)
	end
	#
	num_runs = 1 # Averaging over instances of the sampling pattern
	sampling = MRITrajectories.rand(Int, () -> (rand(), rand()), (num_σ * prod(shape)), shape, num_σ, num_σ)
	sampling = MRIRecon.in_chronological_order(MRITrajectories.swap_duplicates_dynamic(MRIRecon.split_sampling(sampling, num_time)))
	mask = sum(dropdims(MRIRecon.sampling_mask(sampling, shape, num_time); dims=1); dims=3)
	@assert all(mask .== num_σ)
	lr_inversion_recon = reconstruct(sampling)
	jldsave("$(name)_$(num_time)_recon.jld2"; recon=lr_inversion_recon)
	return
end
for (name, path, t0) in dictionary_paths
	simulate(name, path, t0)
end

