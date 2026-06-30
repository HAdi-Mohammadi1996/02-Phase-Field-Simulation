using JLD2
using MAT
using Printf

function load_volume_preserving_function()
    utility_path = joinpath(@__DIR__, "utils", "reconstruct_volume_preserving.jl")
    code = read(utility_path, String)
    code = replace(code, r"(?m)^main\(\)\s*$" => "")
    include_string(Main, code, utility_path)
end

load_volume_preserving_function()

function parse_args(args)
    isempty(args) && error("usage: julia regenerate_sample_mat_volume_preserving.jl SAMPLE_ID [--start N] [--end N]")

    sample_id = parse(Int, first(args))
    start_step = 0
    end_step = nothing

    i = 2
    while i <= length(args)
        if args[i] == "--start"
            i += 1
            start_step = parse(Int, args[i])
        elseif args[i] == "--end"
            i += 1
            end_step = parse(Int, args[i])
        else
            error("unknown argument: $(args[i])")
        end
        i += 1
    end

    return sample_id, start_step, end_step
end

function write_mat_atomic(path::AbstractString, volume)
    temp_path = path * ".tmp"
    isfile(temp_path) && rm(temp_path; force=true)

    matopen(temp_path, "w") do file
        write(file, "C", volume)
    end

    mv(temp_path, path; force=true)
end

function checkpoint_step(path::AbstractString)
    name = splitext(basename(path))[1]
    startswith(name, "t") || error("checkpoint filename does not start with 't': $path")
    return parse(Int, name[2:end])
end

function regenerate_sample(sample_id::Integer, start_step::Integer, end_step)
    initial_mat_path = joinpath(raw"D:\Hadi\SharedData\InitialMatData", "$(sample_id).mat")
    checkpoint_dir = joinpath(raw"D:\Hadi\SharedData\PhaseFieldResults", "$(sample_id)", "results")
    output_dir = joinpath(raw"D:\Hadi\SharedData\PhaseFieldResults", "$(sample_id)", "mat")

    isfile(initial_mat_path) || error("missing original MAT file: $initial_mat_path")
    isdir(checkpoint_dir) || error("missing checkpoint directory: $checkpoint_dir")
    mkpath(output_dir)

    original_phase = matread(initial_mat_path)["C"]
    ysz_mask = original_phase .== 2
    ni_voxel_count = count(==(1), original_phase)

    checkpoint_files = sort(filter(
        path -> endswith(lowercase(path), ".jld2"),
        readdir(checkpoint_dir; join=true),
    ))

    if end_step === nothing
        end_step = maximum(checkpoint_step, checkpoint_files)
    end

    @printf(
        "Sample %d: Ni=%d, YSZ=%d, pore=%d, total=%d, steps=%d:%d\n",
        sample_id,
        ni_voxel_count,
        count(ysz_mask),
        length(original_phase) - ni_voxel_count - count(ysz_mask),
        length(original_phase),
        start_step,
        end_step,
    )
    flush(stdout)

    for checkpoint_path in checkpoint_files
        step = checkpoint_step(checkpoint_path)
        start_step <= step <= end_step || continue

        output_name = splitext(basename(checkpoint_path))[1] * ".mat"
        output_path = joinpath(output_dir, output_name)

        keys_in_file = jldopen(checkpoint_path, "r") do file
            collect(keys(file))
        end

        if "phi" in keys_in_file
            phi = JLD2.load(checkpoint_path, "phi")
            phase_volume, cutoff = reconstruct_volume_preserving(phi, ysz_mask, ni_voxel_count)
            write_mat_atomic(output_path, phase_volume)
            @printf("step %06d: reconstructed from phi, cutoff=%.10g\n", step, cutoff)
        elseif step == 0
            write_mat_atomic(output_path, Int32.(original_phase))
            @printf("step %06d: checkpoint has no phi, wrote original C volume\n", step)
        else
            error("checkpoint has no phi key: $checkpoint_path")
        end

        flush(stdout)
        GC.gc()
    end
end

function main(args=ARGS)
    sample_id, start_step, end_step = parse_args(args)
    regenerate_sample(sample_id, start_step, end_step)
end

main()
