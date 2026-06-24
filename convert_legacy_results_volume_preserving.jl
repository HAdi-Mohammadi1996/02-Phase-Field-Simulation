using JLD2
using MAT
using Printf

# Convert the two old, whole-solution JLD2 files into the newer per-timestep
# checkpoint layout, then write volume-preserving reconstructed MAT volumes.

const DEFAULT_SAMPLE_IDS = (38, 105)
const INITIAL_MAT_DIR = raw"D:\Hadi\SharedData\InitialMatData"
const PHASE_FIELD_RESULTS_DIR = raw"D:\Hadi\SharedData\PhaseFieldResults"

const ORIGINAL_PHASE_KEY = "C"
const CHECKPOINT_FIELD_KEY = "phi"
const OUTPUT_PHASE_KEY = "C"
const DEFAULT_SAVE_INTERVAL = 50.0
const DEFAULT_OVERWRITE_EXISTING = false

function reconstruct_volume_preserving(phi, ysz_mask, ni_voxel_count)
    size(phi) == size(ysz_mask) ||
        throw(DimensionMismatch("phi and the original YSZ mask have different sizes"))
    all(isfinite, phi) ||
        throw(ArgumentError("phi contains NaN or Inf values"))

    available_voxels = count(!, ysz_mask)
    0 <= ni_voxel_count <= available_voxels ||
        throw(ArgumentError("requested Ni count is outside the non-YSZ domain"))

    phase_volume = fill(Int32(3), size(phi))
    phase_volume[ysz_mask] .= Int32(2)

    ni_voxel_count == 0 && return phase_volume, Inf

    if ni_voxel_count == available_voxels
        phase_volume[.!ysz_mask] .= Int32(1)
        return phase_volume, -Inf
    end

    non_ysz_values = Vector{eltype(phi)}(undef, available_voxels)
    position = 0
    @inbounds for index in eachindex(phi, ysz_mask)
        if !ysz_mask[index]
            position += 1
            non_ysz_values[position] = phi[index]
        end
    end
    cutoff = partialsort!(non_ysz_values, ni_voxel_count; rev=true)

    selected = 0
    @inbounds for index in eachindex(phi, ysz_mask)
        if !ysz_mask[index] && phi[index] > cutoff
            phase_volume[index] = Int32(1)
            selected += 1
        end
    end

    remaining = ni_voxel_count - selected
    @inbounds for index in eachindex(phi, ysz_mask)
        remaining == 0 && break
        if !ysz_mask[index] && phi[index] == cutoff
            phase_volume[index] = Int32(1)
            remaining -= 1
        end
    end

    remaining == 0 ||
        error("failed to select the requested number of Ni voxels")

    return phase_volume, cutoff
end

function read_reloffset(bytes::Vector{UInt8}, offset0::Integer)
    0 <= offset0 <= length(bytes) - 8 ||
        throw(BoundsError(bytes, offset0 + 1:offset0 + 8))

    value = zero(UInt64)
    @inbounds for shift in 0:7
        value |= UInt64(bytes[offset0 + shift + 1]) << (8 * shift)
    end
    return JLD2.RelOffset(value)
end

function solution_field_ref(file, key::AbstractString, field::Symbol)
    dataset = JLD2.get_dataset(file, String(key))
    dataset.datatype isa JLD2.SharedDatatype ||
        throw(ArgumentError("legacy solution key '$key' is not a stored Julia object"))

    datatype, _ = JLD2.read_shared_datatype(file, dataset.datatype)
    index = findfirst(==(field), datatype.names)
    index === nothing &&
        throw(KeyError("field '$field' was not found in legacy solution key '$key'"))

    return read_reloffset(dataset.layout.data, datatype.offsets[index])
end

function open_legacy_timesteps(file)
    keys_in_file = collect(keys(file))
    println("  keys: ", join(keys_in_file, ", "))

    if haskey(file, "sol")
        u_ref = solution_field_ref(file, "sol", :u)
        t_ref = solution_field_ref(file, "sol", :t)
        fields = JLD2.ArrayDataset(JLD2.get_dataset(file, u_ref))
        times = JLD2.read_dataset(JLD2.get_dataset(file, t_ref))
        return fields, times
    end

    if haskey(file, "single_stored_object")
        dataset = JLD2.get_dataset(file, "single_stored_object")
        fields = JLD2.ArrayDataset(dataset)
        times = collect(0.0:DEFAULT_SAVE_INTERVAL:DEFAULT_SAVE_INTERVAL * (length(fields) - 1))
        return fields, times
    end

    error("legacy file does not contain 'sol' or 'single_stored_object'")
end

function atomic_write_jld2(path::AbstractString, phi, step::Integer, time::Real)
    temp_path = path * ".tmp"
    isfile(temp_path) && rm(temp_path; force=true)

    jldopen(temp_path, "w"; compress=ZstdFilter(9)) do file
        file[CHECKPOINT_FIELD_KEY] = phi
        file["step"] = step
        file["time"] = Float64(time)
    end

    mv(temp_path, path; force=true)
end

function atomic_write_mat(path::AbstractString, phase_volume)
    temp_path = path * ".tmp"
    isfile(temp_path) && rm(temp_path; force=true)

    matopen(temp_path, "w") do file
        write(file, OUTPUT_PHASE_KEY, phase_volume)
    end

    mv(temp_path, path; force=true)
end

function load_original_phase(sample_id::Integer)
    original_mat_path = joinpath(INITIAL_MAT_DIR, "$(sample_id).mat")
    isfile(original_mat_path) ||
        error("original .mat file does not exist: $original_mat_path")

    original_data = matread(original_mat_path)
    haskey(original_data, ORIGINAL_PHASE_KEY) ||
        error("original .mat file does not contain variable '$ORIGINAL_PHASE_KEY'")

    return original_data[ORIGINAL_PHASE_KEY]
end

function convert_sample(sample_id::Integer; dry_run::Bool=false, overwrite::Bool=DEFAULT_OVERWRITE_EXISTING)
    legacy_path = joinpath(PHASE_FIELD_RESULTS_DIR, "$(sample_id).jld2")
    isfile(legacy_path) ||
        error("legacy .jld2 file does not exist: $legacy_path")

    sample_dir = joinpath(PHASE_FIELD_RESULTS_DIR, "$(sample_id)")
    checkpoint_dir = joinpath(sample_dir, "results")
    mat_dir = joinpath(sample_dir, "mat")
    mkpath(checkpoint_dir)
    mkpath(mat_dir)

    println()
    println("Sample $sample_id")
    println("  legacy:  ", legacy_path)
    println("  results: ", checkpoint_dir)
    println("  mat:     ", mat_dir)

    original_phase = load_original_phase(sample_id)
    ysz_mask = original_phase .== 2
    ni_voxel_count = count(==(1), original_phase)

    @printf(
        "  original volume: Ni=%d, YSZ=%d, pore=%d, total=%d\n",
        ni_voxel_count,
        count(ysz_mask),
        length(original_phase) - ni_voxel_count - count(ysz_mask),
        length(original_phase),
    )

    jldopen(legacy_path, "r") do legacy_file
        fields, times = open_legacy_timesteps(legacy_file)
        length(fields) == length(times) ||
            error("number of saved fields ($(length(fields))) does not match number of times ($(length(times)))")

        println("  timesteps: ", length(fields))
        println("  first time: ", first(times), ", last time: ", last(times))

        if dry_run
            first_phi = fields[1]
            println("  first field: ", typeof(first_phi), ", size=", size(first_phi), ", extrema=", extrema(first_phi))
            return nothing
        end

        for i in eachindex(fields)
            step = i - 1
            checkpoint_name = "t$(lpad(step, 6, '0')).jld2"
            mat_name = "t$(lpad(step, 6, '0')).mat"
            checkpoint_path = joinpath(checkpoint_dir, checkpoint_name)
            mat_path = joinpath(mat_dir, mat_name)

            checkpoint_done = isfile(checkpoint_path)
            mat_done = isfile(mat_path)
            if !overwrite && checkpoint_done && mat_done
                @printf("  [%3d/%3d] skipped %s and %s\n", i, length(fields), checkpoint_name, mat_name)
                continue
            end

            phi = fields[i]
            size(phi) == size(ysz_mask) ||
                throw(DimensionMismatch("saved field size $(size(phi)) does not match original size $(size(ysz_mask))"))

            if overwrite || !checkpoint_done
                atomic_write_jld2(checkpoint_path, phi, step, times[i])
            end

            cutoff = NaN
            if overwrite || !mat_done
                phase_volume, cutoff =
                    reconstruct_volume_preserving(phi, ysz_mask, ni_voxel_count)
                atomic_write_mat(mat_path, phase_volume)
                phase_volume = nothing
            end

            @printf(
                "  [%3d/%3d] %s -> %s, %s (time = %.10g, cutoff = %.10g)\n",
                i,
                length(fields),
                basename(legacy_path),
                checkpoint_name,
                mat_name,
                times[i],
                cutoff,
            )

            phi = nothing
            GC.gc()
        end
    end
end

function parse_args(args)
    dry_run = false
    overwrite = DEFAULT_OVERWRITE_EXISTING
    sample_ids = Int[]

    for arg in args
        if arg == "--dry-run"
            dry_run = true
        elseif arg == "--overwrite"
            overwrite = true
        else
            push!(sample_ids, parse(Int, arg))
        end
    end

    isempty(sample_ids) && append!(sample_ids, DEFAULT_SAMPLE_IDS)
    return sample_ids, dry_run, overwrite
end

function main(args=ARGS)
    sample_ids, dry_run, overwrite = parse_args(args)

    println("Converting legacy JLD2 results with volume-preserving reconstruction")
    println("Checkpoint key: '$CHECKPOINT_FIELD_KEY'")
    println("MAT key:        '$OUTPUT_PHASE_KEY'")
    println("Samples:        ", join(sample_ids, ", "))
    println("Dry run:        $dry_run")
    println("Overwrite:      $overwrite")

    for sample_id in sample_ids
        convert_sample(sample_id; dry_run=dry_run, overwrite=overwrite)
    end

    println()
    println("Done.")
end

main()
