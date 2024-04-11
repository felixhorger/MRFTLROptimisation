using PyPlot
using JLD2
import PyPlotTools
import MRFSeqParams
PyPlotTools.set_image()


num_σ = 4
Jiang_TR, Jiang_α = MRFSeqParams.read_pulses("Jiang2014/MRFParams")[6:7]
Jiang_VH = load("Jiang2014/compressed_dictionary.jld2", "transform")[1:num_σ, :]
Zhao_TR, Zhao_α = MRFSeqParams.read_pulses("Zhao2019/MRFParams")[6:7]
Zhao_VH = load("Zhao2019/compressed_dictionary.jld2", "transform")[1:num_σ, :]
Koolstra_TR, Koolstra_α = MRFSeqParams.read_pulses("Koolstra2018/MRFParams")[6:7]
Koolstra_VH = load("Koolstra2018/compressed_dictionary.jld2", "transform")[1:num_σ, :]
FastCycle_TR, FastCycle_α = MRFSeqParams.read_pulses("../../../20230528_LongDictWithRecovery/optimised")[6:7]
FastCycle_VH = load("../../../20230528_LongDictWithRecovery/compressed_dictionary.jld2", "transform")[1:num_σ, :]
num_time = 1024
Fourier_VH = Matrix{ComplexF64}(undef, num_σ, num_time)
for i = 1:num_σ
	Fourier_VH[i, :] .= exp.(im * 2π / num_time * (i-1) * (0:num_time-1)) ./ sqrt(num_time)
end
jldsave("Fourier/Fourier.jld2"; transform=Fourier_VH)

for (name, path) in (
	("Jiang", "../schedules/Jiang2014/compressed_dictionary.jld2"),
	("Zhao", "../schedules/Zhao2019/compressed_dictionary.jld2"),
	("Koolstra", "../schedules/Koolstra2018/compressed_dictionary.jld2"),
	("FastCycle", "../schedules/FastCycle/compressed_dictionary.jld2")
)
	s = load(path, "singular_values")
	println("$name: $(sqrt(sum(abs2, s[num_σ+1:end]) / sum(abs2, s)))")
end

Jiang_VH[1, :] .*= -1.5
Jiang_VH[2, :] .*= -1.0
Zhao_VH[1, :] .*= -2.0 
Zhao_VH[2, :] .*= -1.0 
Koolstra_VH[1, :] .*= -2.0 
FastCycle_VH[1, :] .*= -2.0 
FastCycle_VH[2, :] .*= -1.0 

grid_spec = Dict("width_ratios" => [1, 1, 1, 1, 1], "height_ratios" => [0.4, 0.3, 0.65])
fig, axs = plt.subplots(3, 5; gridspec_kw=grid_spec, sharex="col", sharey="row", figsize=(1.4 * PyPlotTools.latex_column, 2.85))
plt.subplots_adjust(hspace=0.2, wspace=0.15)
axs[1, 1].set_title("Jiang")
axs[1, 2].set_title("Zhao")
axs[1, 3].set_title("Koolstra")
axs[1, 4].set_title("Fast Cycle")
axs[1, 5].set_title("Fourier")
axs[1, 1].plot(1:length(Jiang_α)-1, rad2deg.(Jiang_α[2:end]))
axs[1, 2].plot(1:length(Zhao_α)-1, rad2deg.(Zhao_α[2:end]))
axs[1, 3].plot(1:length(Koolstra_α)-1, rad2deg.(Koolstra_α[2:end]))
axs[1, 4].plot(1:length(FastCycle_α)-1, rad2deg.(FastCycle_α[2:end]))
axs[1, 5].set_axis_off()
axs[2, 1].plot(1:length(Jiang_TR)-2, Jiang_TR[2:end-1])
axs[2, 2].plot(1:length(Zhao_TR)-2, Zhao_TR[2:end-1])
axs[2, 3].plot(1:length(Koolstra_TR)-2, Koolstra_TR[2:end-1])
axs[2, 4].plot(1:length(FastCycle_TR)-2, FastCycle_TR[2:end-1])
axs[2, 5].set_axis_off()
axs[1, 1].set_xticks([1, 500, length(Jiang_TR)])
axs[1, 2].set_xticks([1, 200, length(Zhao_TR)])
axs[1, 3].set_xticks([1, 120, length(Koolstra_TR)])
axs[1, 4].set_xticks([1, 420, length(FastCycle_TR)])
axs[1, 5].set_xticks([1, 500, size(Fourier_VH, 2)])
offset = reverse(0:0.25:0.75)'
lengths = length.((Jiang_TR, Zhao_TR, Koolstra_TR, FastCycle_TR, 1:num_time))
for i = 1:num_σ, j = 1:5
	axs[3, j].plot(1:lengths[j]-2, fill(offset[i], lengths[j]-2); color="black", linewidth=0.5)
end
axs[3, 1].plot(1:length(Jiang_TR)-2, offset .+ 2 .* Jiang_VH[:, 2:end-1]')
axs[3, 2].plot(1:length(Zhao_TR)-2, offset .+ Zhao_VH[:, 2:end-1]')
axs[3, 3].plot(1:length(Koolstra_TR)-2, offset .+ Koolstra_VH[:, 2:end-1]')
axs[3, 4].plot(1:length(FastCycle_TR)-2, offset .+ 1.75 .* FastCycle_VH[:, 2:end-1]')
for i = 1:num_σ
	axs[3, 5].plot(1:num_time, offset[i] .+ 2 .* real.(Fourier_VH[i, :]); label="$i")
end
axs[1, 1].set_yticks([0, 35, 70])
axs[1, 5].text(500, 50, "no schedule available"; ha="center")
foreach(i -> axs[3, i].set_xlabel("\$T_R\$-index"), 1:5)
foreach(i -> axs[1, i].set_ylabel("\$\\alpha\$ [\\SI{}{\\degree}]"), 1:1)
foreach(i -> axs[1, i].set_ylim([0, 72]), 1:5)
foreach(i -> axs[2, i].set_ylabel("\$T_R\$ [\\SI{}{\\milli\\second}]"), 1:1)
foreach(i -> axs[2, i].set_ylim([3, 17]), 1:5)
foreach(i -> axs[3, i].set_ylim([-0.2, 1.0]), 1:5)
foreach(i -> axs[3, i].tick_params(left=false, labelleft=false), 1:5)
foreach(i -> axs[3, i].set_ylabel("Singular vectors"), 1:1)
axs[3, 5].legend(loc="center", ncol=2, columnspacing=1, handlelength=1.5, bbox_to_anchor=(0.53, 1.375), title="Singular vectors")
plt.savefig("schedules.eps")

