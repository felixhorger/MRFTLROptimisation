function sampling_mask(sampling_indices, shape, num_time)
	split_sampling = MRIRecon.split_sampling(sampling_indices, num_time)
	split_sampling = MRITrajectories.swap_duplicates_dynamic!(split_sampling)
	split_sampling = MRITrajectories.sort_constantL2distance!(split_sampling, 10.0)
	sampling_indices = MRIRecon.in_chronological_order(split_sampling)
	mask = MRIRecon.sampling_mask(sampling_indices, shape, num_time)
	return mask, sampling_indices
end
function radial_density()
	ϕ = 2π * rand()
	r = 0.5 * rand()
	sine, cosine = sincos(ϕ)
	x = r * cosine + 0.5
	y = r * sine + 0.5
	return x, y
end
